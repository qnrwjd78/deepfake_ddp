"""Standalone DevDet-style components for GatedDualDetector.

This module intentionally does not modify model.py.  The feature adapter below
repeats GatedDualDetector.forward up to (but excluding) its final classifier.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.5):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class FFDevGenerator(nn.Module):
    """CDA-inspired six-block image reconstruction generator.

    It returns a signed residual in [-1, 1].  Image composition and clamping
    are deliberately outside the network so both direct and scaled dose rules
    can be reproduced.
    """

    def __init__(self, base_channels: int = 64, blocks: int = 6, dropout: float = 0.5):
        super().__init__()
        c = int(base_channels)
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3), nn.Conv2d(3, c, 7, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, 2 * c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * c), nn.ReLU(inplace=True),
            nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * c), nn.ReLU(inplace=True),
        ]
        layers += [ResidualBlock(4 * c, dropout) for _ in range(int(blocks))]
        layers += [
            nn.ConvTranspose2d(4 * c, 2 * c, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(2 * c), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2 * c, c, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3), nn.Conv2d(c, 3, 7), nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def develop_image(
    image: torch.Tensor,
    delta: torch.Tensor,
    dose: float | torch.Tensor,
    clamp: bool = True,
) -> torch.Tensor:
    if image.shape != delta.shape:
        raise ValueError(f"image/delta shape mismatch: {tuple(image.shape)} vs {tuple(delta.shape)}")
    if torch.is_tensor(dose) and dose.numel() != 1:
        if dose.shape[0] != image.shape[0] or any(size != 1 for size in dose.shape[1:]):
            raise ValueError(f"dose must be scalar or one value per image, got {tuple(dose.shape)}")
        dose = dose.reshape(image.shape[0], 1, 1, 1)
    result = image + dose * delta
    return result.clamp(0.0, 1.0) if clamp else result


def total_variation(image: torch.Tensor) -> torch.Tensor:
    vertical = image[:, :, 1:, :-1] - image[:, :, :-1, :-1]
    horizontal = image[:, :, :-1, 1:] - image[:, :, :-1, :-1]
    return torch.sqrt(vertical.square() + horizontal.square() + 1e-12).mean()


def resolve_dose_scale(daft_config: dict, ffdev_epsilon: float) -> float:
    """Resolve the paper's ambiguous direct-vs-scaled adaptive dose rule."""
    mode = str(daft_config.get("dose_mode", "direct")).lower()
    if mode == "direct":
        return 1.0
    if mode == "scaled":
        return float(daft_config.get("base_epsilon", ffdev_epsilon))
    raise ValueError(f"Unknown dose_mode={mode!r}; expected 'direct' or 'scaled'")


def gated_dual_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return the exact gated feature consumed by ``model.fc``."""
    required = ("g_net", "l_net", "g_ln", "l_ln", "gate_net", "gate_temp", "fc")
    if any(not hasattr(model, name) for name in required):
        raise TypeError("DoseDict adapter requires a GatedDualDetector")
    g = model.g_ln(model._branch(
        model.g_net, x, model.g_size, model.g_mean, model.g_std,
        model.g_ft not in ("none", ""),
    ))
    l = model.l_ln(model._branch(
        model.l_net, x, model.l_size, model.l_mean, model.l_std,
        model.l_ft not in ("none", ""),
    ))
    features = [g, l]
    if model.fftcut is not None:
        features.append(model.fft_ln(model.fftcut(x)))
    joined = torch.cat(features, dim=1)
    gates = torch.softmax(model.gate_net(joined) / model.gate_temp, dim=1)
    if model.gate_mode == "none":
        return joined
    return torch.cat(
        [feature * gates[:, index:index + 1] for index, feature in enumerate(features)], dim=1
    )


def gated_dual_logits_and_features(model: nn.Module, x: torch.Tensor):
    feature = gated_dual_features(model, x)
    return model.fc(feature), feature


def soft_threshold(x: torch.Tensor, threshold: float | torch.Tensor) -> torch.Tensor:
    return x.sign() * (x.abs() - threshold).clamp_min(0.0)


@torch.no_grad()
def sparse_code(
    features: torch.Tensor,
    dictionary: torch.Tensor,
    lasso_lambda: float,
    iterations: int,
    gram: torch.Tensor | None = None,
    lipschitz: torch.Tensor | None = None,
) -> torch.Tensor:
    """ISTA for min_a 0.5 ||aD-z||^2 + lambda ||a||_1."""
    # Infinity norm of D D^T is an upper bound on its spectral radius and is
    # considerably tighter than the sum of all atom energies.
    if gram is None:
        gram = dictionary @ dictionary.t()
    if lipschitz is None:
        lipschitz = gram.abs().sum(dim=1).max().clamp_min(1e-8)
    step = 1.0 / lipschitz
    code = features.new_zeros((features.shape[0], dictionary.shape[0]))
    projected = features @ dictionary.t()
    for _ in range(int(iterations)):
        gradient = code @ gram - projected
        code = soft_threshold(code - step * gradient, lasso_lambda * step)
    return code


@dataclass
class DoseDictionary:
    atoms: torch.Tensor
    lasso_lambda: float
    ista_iterations: int
    error_low: float
    error_high: float
    l2_normalize: bool = True
    _gram: torch.Tensor | None = field(default=None, init=False, repr=False)
    _lipschitz: torch.Tensor | None = field(default=None, init=False, repr=False)
    _cache_key: tuple | None = field(default=None, init=False, repr=False)

    def _operators(self, atoms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = (atoms.device, atoms.dtype, atoms.data_ptr(), atoms._version)
        if self._cache_key != key:
            self._gram = atoms @ atoms.t()
            self._lipschitz = self._gram.abs().sum(dim=1).max().clamp_min(1e-8)
            self._cache_key = key
        return self._gram, self._lipschitz

    def prepare(self, features: torch.Tensor) -> torch.Tensor:
        features = features.float()
        return F.normalize(features, dim=1) if self.l2_normalize else features

    @torch.no_grad()
    def error(self, features: torch.Tensor) -> torch.Tensor:
        no_amp = (
            torch.autocast(device_type="cuda", enabled=False)
            if features.is_cuda else contextlib.nullcontext()
        )
        with no_amp:
            features = self.prepare(features)
            atoms = self.atoms.to(device=features.device, dtype=torch.float32)
            gram, lipschitz = self._operators(atoms)
            code = sparse_code(
                features, atoms, self.lasso_lambda, self.ista_iterations,
                gram=gram, lipschitz=lipschitz,
            )
            return (features - code @ atoms).norm(dim=1)

    @torch.no_grad()
    def dose(self, features: torch.Tensor) -> torch.Tensor:
        error = self.error(features)
        width = max(self.error_high - self.error_low, 1e-8)
        # Robust min-max normalization of 1-error: low error -> high dose.
        return ((self.error_high - error) / width).clamp(0.0, 1.0)

    def state_dict(self) -> dict:
        return {
            "atoms": self.atoms.cpu(),
            "lasso_lambda": self.lasso_lambda,
            "ista_iterations": self.ista_iterations,
            "error_low": self.error_low,
            "error_high": self.error_high,
            "l2_normalize": self.l2_normalize,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "DoseDictionary":
        return cls(**state)


def fit_dictionary(
    features: torch.Tensor,
    atoms: int = 256,
    lasso_lambda: float = 0.05,
    ista_iterations: int = 40,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 0.1,
    l2_normalize: bool = True,
    seed: int = 0,
    device: str | torch.device = "cuda",
) -> tuple[DoseDictionary, list[float]]:
    """Fit atoms by alternating ISTA coding and atom SGD; no sklearn required."""
    device = torch.device(device)
    z = features.float()
    if l2_normalize:
        z = F.normalize(z, dim=1)
    if len(z) < atoms:
        raise ValueError(f"Need at least {atoms} features, got {len(z)}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randperm(len(z), generator=generator)[:atoms]
    dictionary = nn.Parameter(F.normalize(z[initial].to(device), dim=1))
    optimizer = torch.optim.SGD([dictionary], lr=learning_rate, momentum=0.9)
    losses: list[float] = []
    batches_per_epoch = math.ceil(len(z) / batch_size)
    progress = tqdm(
        total=int(epochs) * batches_per_epoch,
        desc="DoseDict fitting",
        unit="batch",
        dynamic_ncols=True,
    )
    for epoch in range(int(epochs)):
        order = torch.randperm(len(z), generator=generator)
        total = 0.0
        seen = 0
        for start in range(0, len(z), batch_size):
            batch = z[order[start:start + batch_size]].to(device)
            with torch.no_grad():
                code = sparse_code(batch, dictionary.detach(), lasso_lambda, ista_iterations)
            reconstruction = code @ dictionary
            loss = 0.5 * (reconstruction - batch).square().sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                dictionary.copy_(F.normalize(dictionary, dim=1))
            total += loss.item() * len(batch)
            seen += len(batch)
            progress.update(1)
            progress.set_postfix(
                epoch=f"{epoch + 1}/{int(epochs)}",
                loss=f"{total / max(seen, 1):.6f}",
                refresh=False,
            )
        losses.append(total / max(seen, 1))
    progress.close()
    fitted_atoms = dictionary.detach()
    errors = []
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            batch = z[start:start + batch_size].to(device)
            code = sparse_code(batch, fitted_atoms, lasso_lambda, ista_iterations)
            errors.append((batch - code @ fitted_atoms).norm(dim=1).cpu())
    errors_t = torch.cat(errors)
    low, high = torch.quantile(errors_t, torch.tensor([0.05, 0.95])).tolist()
    result = DoseDictionary(
        atoms=fitted_atoms.cpu(), lasso_lambda=float(lasso_lambda),
        ista_iterations=int(ista_iterations), error_low=float(low), error_high=float(high),
        l2_normalize=bool(l2_normalize),
    )
    return result, losses


class DevDetInference(nn.Module):
    def __init__(
        self, detector: nn.Module, generator: FFDevGenerator,
        dictionary: DoseDictionary, base_epsilon: float = 1.0,
    ):
        super().__init__()
        self.detector = detector
        self.generator = generator
        self.dictionary = dictionary
        self.base_epsilon = float(base_epsilon)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feature = gated_dual_features(self.detector, image)
            dose = self.dictionary.dose(feature) * self.base_epsilon
            delta = self.generator(image)
            developed = develop_image(image, delta, dose)
        return self.detector(developed)
