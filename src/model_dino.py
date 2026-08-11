"""baseline_DINO model (separate from model.py; CLIP baseline untouched).

  global = DINOv3 (LoRA), sees the WIDE view (loose crop / context)   -> forward arg x_wide
  local  = Swin-B  (LoRA), sees the TIGHT aligned face crop            -> forward arg x
  FFT branch on the tight crop. Fused by a per-branch SIGMOID gate (independent, not zero-sum).
  Optimizer SAM (base AdamW recommended for PEFT). Only LoRA + norms + gate + fc train.

forward(x, x_wide=None): local/FFT use x (tight); global uses x_wide (falls back to x if None,
so single-view eval still runs, degraded)."""
import torch
from torch import nn
from torch.nn import functional as F
from timm import create_model
from functools import partial

from utils.sam import SAM
from model import param_groups_weight_decay, FFTCutoffEmbed


class GatedDualDINO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg['in_chans'] == 3
        self.g_size = cfg.get('global_img_size', 224)
        self.l_size = cfg.get('local_img_size', 384)
        self.g_net = create_model(cfg.get('global_backbone', 'vit_large_patch16_dinov3.lvd1689m'), pretrained=True, num_classes=0)
        self.l_net = create_model(cfg.get('local_backbone', 'swin_base_patch4_window12_384'), pretrained=True, num_classes=0)
        gdim, ldim = self.g_net.num_features, self.l_net.num_features            # capture BEFORE LoRA wrap
        for tag, net in (('g', self.g_net), ('l', self.l_net)):                  # mean/std from raw timm cfg
            self.register_buffer(f'{tag}_mean', torch.tensor(net.default_cfg['mean']).view(1, -1, 1, 1))
            self.register_buffer(f'{tag}_std',  torch.tensor(net.default_cfg['std']).view(1, -1, 1, 1))
        r = int(cfg.get('lora_r', 32)); alpha = int(cfg.get('lora_alpha', 64)); drop = float(cfg.get('lora_dropout', 0.05))
        self.g_net = self._lora(self.g_net, r, alpha, drop)                      # freezes base, trains adapters
        self.l_net = self._lora(self.l_net, r, alpha, drop)
        self.g_ln = nn.LayerNorm(gdim); self.l_ln = nn.LayerNorm(ldim)

        self.fftcut = None; fdim = 0
        if cfg.get('fft_cutoffs') is not None:
            fdim = cfg['fft_out_dim']
            self.fftcut = FFTCutoffEmbed(cfg['fft_cutoffs'], cfg['fft_ring_bw'], fdim)
            self.fft_ln = nn.LayerNorm(fdim)
        self.n_br = 2 + (1 if self.fftcut else 0)
        self.gate_mode = str(cfg.get('gate_mode', 'sigmoid'))                    # 'sigmoid' | 'gated'(softmax) | 'none'
        self.gate_temp = cfg.get('gate_temp', 1.0)
        self.gate_net = nn.Sequential(nn.Linear(gdim+ldim+fdim, cfg['gate_hidden_dim']), nn.ReLU(),
                                      nn.Linear(cfg['gate_hidden_dim'], self.n_br))
        self.fc = nn.Linear(gdim+ldim+fdim, 2)

        cfg_opt = cfg['optimizer']; assert cfg_opt['name'] == 'SAM'; rho = cfg_opt['rho']; bo = cfg_opt['base']
        if bo['name'] == 'AdamW':
            pg = param_groups_weight_decay(self, weight_decay=bo.get('weight_decay', 0.01))
            base = partial(torch.optim.AdamW, lr=bo['lr'], betas=tuple(bo.get('betas', [0.9, 0.999])),
                           eps=bo.get('eps', 1e-8), weight_decay=bo.get('weight_decay', 0.01))
            self.optimizer = SAM(pg, base, rho=rho)
        elif bo['name'] == 'SGD':
            tr = [p for p in self.parameters() if p.requires_grad]
            self.optimizer = SAM(tr, torch.optim.SGD, rho=rho, lr=bo['lr'], momentum=bo['momentum'])
        else: raise NotImplementedError

        nn.init.zeros_(self.gate_net[-1].weight); nn.init.zeros_(self.gate_net[-1].bias)
        n_tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"GatedDualDINO: global={cfg.get('global_backbone')}(lora,wide{self.g_size}) "
              f"local={cfg.get('local_backbone')}(lora,tight{self.l_size}) fft={self.fftcut is not None} "
              f"gate={self.gate_mode} trainable={n_tr/1e6:.2f}M")

    @staticmethod
    def _lora(net, r, alpha, drop):
        from peft import LoraConfig, get_peft_model
        # NO task_type: it's a vision model (timm). A task_type (e.g. FEATURE_EXTRACTION) makes
        # PeftModel inject text kwargs like input_ids into forward. 'qkv' matches both ViT(DINOv3) & Swin attn.
        lc = LoraConfig(r=r, lora_alpha=alpha, target_modules=["qkv"], lora_dropout=drop, bias="none")
        return get_peft_model(net, lc)

    def _branch(self, net, x, size, mean, std):
        if size is not None and x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
        x = (x - mean) / std
        return net(x)                              # LoRA: base frozen, adapters + norms train

    def forward(self, x, x_wide=None):
        if x_wide is None: x_wide = x              # single-view fallback (degraded eval)
        g = self.g_ln(self._branch(self.g_net, x_wide, self.g_size, self.g_mean, self.g_std))   # global = WIDE
        l = self.l_ln(self._branch(self.l_net, x,      self.l_size, self.l_mean, self.l_std))   # local  = TIGHT
        feats = [g, l]
        if self.fftcut is not None: feats.append(self.fft_ln(self.fftcut(x)))                    # FFT on tight
        cat = torch.cat(feats, dim=1)
        if self.gate_mode == 'none':
            fused = cat
        elif self.gate_mode == 'sigmoid':
            gth = torch.sigmoid(self.gate_net(cat))                                              # independent per-branch
            fused = torch.cat([feats[i] * gth[:, i:i+1] for i in range(len(feats))], dim=1)
        else:  # 'gated' = softmax (zero-sum)
            gth = torch.softmax(self.gate_net(cat) / self.gate_temp, dim=1)
            fused = torch.cat([feats[i] * gth[:, i:i+1] for i in range(len(feats))], dim=1)
        return self.fc(fused)
