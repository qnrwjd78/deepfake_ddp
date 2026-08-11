import torch
from torch import nn
from torch.nn import functional as F

from timm import create_model
from functools import partial

from utils.sam import SAM


def param_groups_weight_decay(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # bias 및 모든 1D 파라미터(= LayerNorm 등)는 no-decay
        if p.ndim == 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class FFTCutoffEmbed(nn.Module):
    """
    - cutoffs: Nyquist=1로 정규화한 반지름 비율(tuple). 예: (0.25, 0.40)
    - ring_bw: 컷오프 주변 링 폭(정규화 반지름 단위). 0이면 비활성.
    - 출력: f (B, out_dim), in_dim=4*K+1, K=len(cutoffs)
      [logE_low, logE_high, high_ratio, ring_ratio]*K + spectral_centroid
    """
    def __init__(self, cutoffs=(0.25, 0.40), ring_bw=0.04, out_dim=256):
        super().__init__()
        self.cutoffs = tuple(float(c) for c in cutoffs)
        self.ring_bw = float(ring_bw)
        in_dim = len(self.cutoffs)*4 + 1
        self.proj = nn.Linear(in_dim, out_dim)

    def _to_gray(self, x):  # x: (B,3,H,W), [0,1]
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        return (0.2989 * r + 0.5870 * g + 0.1140 * b).unsqueeze(1)  # (B,1,H,W)

    def _radial_grid(self, H, W, device):
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r = r / (r.max() + 1e-8)  # 0..1
        return r  # (H,W)

    def forward(self, x):  # x: (B,3,H,W), float/half
        x = x.float()
        gray = self._to_gray(x).clamp(0, 1)
        gray = gray - gray.mean(dim=(2, 3), keepdim=True)

        F2  = torch.fft.fft2(gray)
        F2s = torch.fft.fftshift(F2)
        mag = torch.abs(F2s)
        mag2 = mag * mag

        B, _, H, W = mag2.shape
        r = self._radial_grid(H, W, mag2.device).view(1, 1, H, W)
        total = mag2.sum(dim=(1, 2, 3)) + 1e-8

        feats = []
        bw = max(self.ring_bw, 0.0)
        for c in self.cutoffs:
            c = float(c)
            lp_mask = (r <= c).to(mag2.dtype)
            hp_mask = 1.0 - lp_mask

            lp_e = (mag2 * lp_mask).sum(dim=(1, 2, 3))
            hp_e = (mag2 * hp_mask).sum(dim=(1, 2, 3))

            if bw > 0:
                ring_mask = ((r >= (c - bw)) & (r <= (c + bw))).to(mag2.dtype)
                ring_e = (mag2 * ring_mask).sum(dim=(1, 2, 3))
                ring_ratio = ring_e / total
            else:
                ring_ratio = torch.zeros_like(total)

            feats.extend([
                torch.log1p(lp_e),   # 저주파 로그 에너지
                torch.log1p(hp_e),   # 고주파 로그 에너지
                hp_e / total,        # 고주파 비율
                ring_ratio,          # 링 비율
            ])

        # 스펙트럼 중심(고주파 쏠림)
        num = (mag * r).sum(dim=(1, 2, 3))
        den = mag.sum(dim=(1, 2, 3)) + 1e-8
        centroid = num / den
        feats.append(centroid)
        f = torch.stack(feats, dim=1) # (B, in_dim)
        return self.proj(f) # (B, out_dim)


class FMDetector(nn.Module):
    """Single foundation-model backbone, no gate / no FFT — the paper-faithful
    'PEFT a frozen foundation model' recipe (Proposed/GenD, CLIP-LN). Configurable:
      backbone:           any timm model name (DINOv3 / SigLIP / CLIP ViT-L14 / ...)
      finetune:           'linear' (head only) | 'ln' (LayerNorm affine + head)
                          | 'ln_gamma' (LN + LayerScale) | 'last_k' | 'full'
      feat_l2norm:        L2-normalize feature + learnable logit scale (CLIP-style)
      backbone_img_size:  resize SBI crops to this before the backbone (else pass-through)
    """
    def __init__(self, cfg):
        super(FMDetector, self).__init__()
        assert cfg['in_chans'] == 3
        self.backbone_name = cfg.get('backbone', 'vit_base_patch16_dinov3.lvd1689m')
        self.net = create_model(self.backbone_name, pretrained=True, num_classes=0)
        feat_dim = self.net.num_features
        self.bb_size = cfg.get('backbone_img_size', None)  # None -> feed image_size as-is

        self.finetune = str(cfg.get('finetune', 'ln'))
        self._set_trainable(self.finetune, int(cfg.get('last_k', 0)))

        self.feat_l2norm = bool(cfg.get('feat_l2norm', False))
        self.head = nn.Linear(feat_dim, 2)
        if self.feat_l2norm:
            self.logit_scale = nn.Parameter(torch.tensor(float(cfg.get('logit_scale_init', 2.659))))

        cfg_opt = cfg['optimizer']
        assert cfg_opt['name'] == 'SAM'
        rho = cfg_opt['rho']
        cfg_bopt = cfg_opt['base']
        if cfg_bopt['name'] == 'SGD':
            trainable = [p for p in self.parameters() if p.requires_grad]
            self.optimizer = SAM(trainable, torch.optim.SGD, rho=rho,
                                 lr=cfg_bopt['lr'], momentum=cfg_bopt['momentum'])
        elif cfg_bopt['name'] == 'AdamW':
            pg = param_groups_weight_decay(self, weight_decay=cfg_bopt.get('weight_decay', 0.01))
            base_opt = partial(torch.optim.AdamW, lr=cfg_bopt['lr'],
                               betas=tuple(cfg_bopt.get('betas', [0.9, 0.999])),
                               eps=cfg_bopt.get('eps', 1e-8),
                               weight_decay=cfg_bopt.get('weight_decay', 0.01))
            self.optimizer = SAM(pg, base_opt, rho=rho)
        else:
            raise NotImplementedError

        mean = torch.tensor(self.net.default_cfg['mean']).view(1, -1, 1, 1)
        std = torch.tensor(self.net.default_cfg['std']).view(1, -1, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"FMDetector backbone={self.backbone_name} finetune={self.finetune} "
              f"feat_dim={feat_dim} trainable={n_tr/1e6:.3f}M l2={self.feat_l2norm}")

    def _set_trainable(self, mode, last_k=0):
        self.net.requires_grad_(mode == 'full')   # head (added later) stays trainable
        if mode in ('full', 'linear'):
            return
        if mode in ('ln', 'ln_gamma'):
            for n, p in self.net.named_parameters():
                is_ln = ('.norm' in n) or n.startswith('norm') or ('norm_pre' in n) or ('fc_norm' in n)
                if is_ln or (mode == 'ln_gamma' and 'gamma' in n):
                    p.requires_grad_(True)
        elif mode == 'last_k':
            nb = len(self.net.blocks)
            keep = set(range(max(0, nb - last_k), nb))
            for n, p in self.net.named_parameters():
                if n.startswith('blocks.') and int(n.split('.')[1]) in keep:
                    p.requires_grad_(True)
                elif n.startswith('norm'):
                    p.requires_grad_(True)
        else:
            raise ValueError(f"unknown finetune mode: {mode}")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.finetune == 'linear':
            self.net.eval()   # fully frozen backbone -> deterministic
        return self

    def forward(self, x):
        if self.bb_size is not None and x.shape[-1] != self.bb_size:
            x = F.interpolate(x, size=(self.bb_size, self.bb_size), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        if self.finetune == 'linear':
            with torch.no_grad():
                feat = self.net(x)
        else:
            feat = self.net(x)
        if self.feat_l2norm:
            feat = F.normalize(feat, dim=1) * self.logit_scale.exp()
        return self.head(feat)


class GatedDualDetector(nn.Module):
    """Dual global+local backbones + optional FFT, gated, partial fine-tune (no full freeze).
      global_backbone / local_backbone: any timm name (e.g. CLIP-L/14 global, Swin-B local)
      global_finetune / local_finetune:  none|ln|ln_gamma|last_k(+*_last_k)|full
      *_img_size: resize input for that branch (e.g. 224 for CLIP-L/14)
      FFT (fft_cutoffs set): a 3rd GATED branch that also feeds the classifier (not just the gate)."""
    def __init__(self, cfg):
        super(GatedDualDetector, self).__init__()
        assert cfg['in_chans'] == 3
        self.g_ft = str(cfg.get('global_finetune', 'ln'))
        self.l_ft = str(cfg.get('local_finetune', 'full'))
        self.g_net = create_model(cfg.get('global_backbone', 'vit_large_patch14_clip_224.openai'), pretrained=True, num_classes=0)
        self.l_net = create_model(cfg.get('local_backbone', 'swin_base_patch4_window12_384'), pretrained=True, num_classes=0)
        self.g_size = cfg.get('global_img_size', 224)
        self.l_size = cfg.get('local_img_size', None)
        self._set_trainable(self.g_net, self.g_ft, int(cfg.get('global_last_k', 0)))
        self._set_trainable(self.l_net, self.l_ft, int(cfg.get('local_last_k', 0)))
        gdim, ldim = self.g_net.num_features, self.l_net.num_features
        self.g_ln = nn.LayerNorm(gdim); self.l_ln = nn.LayerNorm(ldim)

        self.fftcut = None; fdim = 0
        if cfg.get('fft_cutoffs') is not None:
            fdim = cfg['fft_out_dim']
            self.fftcut = FFTCutoffEmbed(cfg['fft_cutoffs'], cfg['fft_ring_bw'], fdim)
            self.fft_ln = nn.LayerNorm(fdim)
        self.n_br = 2 + (1 if self.fftcut else 0)
        self.gate_temp = cfg['gate_temp']
        self.gate_mode = str(cfg.get('gate_mode', 'gated'))   # 'gated' | 'none' (plain concat)
        self.gate_net = nn.Sequential(nn.Linear(gdim+ldim+fdim, cfg['gate_hidden_dim']), nn.ReLU(),
                                      nn.Linear(cfg['gate_hidden_dim'], self.n_br))
        self.fc = nn.Linear(gdim+ldim+fdim, 2)

        cfg_opt = cfg['optimizer']; assert cfg_opt['name'] == 'SAM'; rho = cfg_opt['rho']; bo = cfg_opt['base']
        if bo['name'] == 'SGD':
            tr = [p for p in self.parameters() if p.requires_grad]
            self.optimizer = SAM(tr, torch.optim.SGD, rho=rho, lr=bo['lr'], momentum=bo['momentum'])
        elif bo['name'] == 'AdamW':
            pg = param_groups_weight_decay(self, weight_decay=bo.get('weight_decay', 0.01))
            base = partial(torch.optim.AdamW, lr=bo['lr'], betas=tuple(bo.get('betas', [0.9, 0.999])),
                           eps=bo.get('eps', 1e-8), weight_decay=bo.get('weight_decay', 0.01))
            self.optimizer = SAM(pg, base, rho=rho)
        else: raise NotImplementedError

        for tag, net in (('g', self.g_net), ('l', self.l_net)):
            self.register_buffer(f'{tag}_mean', torch.tensor(net.default_cfg['mean']).view(1, -1, 1, 1))
            self.register_buffer(f'{tag}_std', torch.tensor(net.default_cfg['std']).view(1, -1, 1, 1))
        nn.init.zeros_(self.gate_net[-1].weight); nn.init.zeros_(self.gate_net[-1].bias)
        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"GatedDual: global={cfg.get('global_backbone')}({self.g_ft}) local={cfg.get('local_backbone')}({self.l_ft}) "
              f"fft={self.fftcut is not None} trainable={n_tr/1e6:.2f}M")

    def _set_trainable(self, net, mode, last_k=0):
        net.requires_grad_(mode == 'full')
        if mode in ('full', 'none', ''): return
        if mode in ('ln', 'ln_gamma'):
            for n, p in net.named_parameters():
                if ('.norm' in n) or n.startswith('norm') or ('norm_pre' in n) or ('fc_norm' in n) or (mode == 'ln_gamma' and 'gamma' in n):
                    p.requires_grad_(True)
        elif mode == 'last_k':
            nb = len(net.blocks); keep = set(range(max(0, nb-last_k), nb))
            for n, p in net.named_parameters():
                if n.startswith('blocks.') and int(n.split('.')[1]) in keep: p.requires_grad_(True)
                elif n.startswith('norm'): p.requires_grad_(True)
        else: raise ValueError(f"bad finetune mode {mode}")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.g_ft in ('none', ''): self.g_net.eval()
        if self.l_ft in ('none', ''): self.l_net.eval()
        return self

    def _branch(self, net, x, size, mean, std, trainable):
        if size is not None and x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
        x = (x - mean) / std
        if trainable: return net(x)
        with torch.no_grad(): return net(x)

    def forward(self, x):
        g = self.g_ln(self._branch(self.g_net, x, self.g_size, self.g_mean, self.g_std, self.g_ft not in ('none','')))
        l = self.l_ln(self._branch(self.l_net, x, self.l_size, self.l_mean, self.l_std, self.l_ft not in ('none','')))
        feats = [g, l]
        if self.fftcut is not None: feats.append(self.fft_ln(self.fftcut(x)))
        gth = torch.softmax(self.gate_net(torch.cat(feats, dim=1)) / self.gate_temp, dim=1)
        if self.gate_mode == 'none':
            fused = torch.cat(feats, dim=1)
        else:
            fused = torch.cat([feats[i] * gth[:, i:i+1] for i in range(len(feats))], dim=1)
        return self.fc(fused)
