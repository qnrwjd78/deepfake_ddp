"""XM-only model adapter.

This module deliberately leaves ``model.py`` untouched.  The subclass has the
same parameters and state-dict keys as ``GatedDualDetector``; it only exposes
the post-gate fused feature and gate weights when the XM trainer asks for them.
"""

import torch
from torch.nn import functional as F

from model import GatedDualDetector


class GatedDualDetectorXM(GatedDualDetector):
    """GatedDualDetector with an opt-in auxiliary training return."""

    def forward(self, x, *, return_aux=False):
        g = self.g_ln(
            self._branch(
                self.g_net,
                x,
                self.g_size,
                self.g_mean,
                self.g_std,
                self.g_ft not in ("none", ""),
            )
        )
        l = self.l_ln(
            self._branch(
                self.l_net,
                x,
                self.l_size,
                self.l_mean,
                self.l_std,
                self.l_ft not in ("none", ""),
            )
        )
        feats = [g, l]
        if self.fftcut is not None:
            feats.append(self.fft_ln(self.fftcut(x)))

        cat = torch.cat(feats, dim=1)
        gate = torch.softmax(self.gate_net(cat) / self.gate_temp, dim=1)
        if self.gate_mode == "none":
            fused = cat
        else:
            fused = torch.cat(
                [feats[i] * gate[:, i : i + 1] for i in range(len(feats))],
                dim=1,
            )

        logits = self.fc(fused)
        if return_aux:
            return logits, {"fused": fused, "gate": gate}
        return logits

