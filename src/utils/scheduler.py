import math
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import _LRScheduler

class LinearDecayLR(_LRScheduler):
    def __init__(self, optimizer, n_epoch, start_decay_ratio=0.75, last_epoch=-1):
        self.start_decay=int(n_epoch * start_decay_ratio)
        self.n_epoch=n_epoch
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        last_epoch = self.last_epoch
        n_epoch=self.n_epoch
        start_decay=self.start_decay
        # per-param-group: scale each group's base_lr by the same decay factor
        # (identical to the old single-group behavior when there is one group).
        lrs = []
        for b_lr in self.base_lrs:
            if last_epoch>start_decay:
                lr=b_lr-b_lr/(n_epoch-start_decay)*(last_epoch-start_decay)
            else:
                lr=b_lr
            lrs.append(lr)
        return lrs

class CosineDecayLR(_LRScheduler):
    def __init__(self, optimizer, n_epoch, min_lr=2e-6, last_epoch=-1):
        self.n_epoch = n_epoch
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch + 1
        progress = min(max(t / self.n_epoch, 0.0), 1.0)
        lrs = []
        for b_lr in self.base_lrs:
            eta_min = min(self.min_lr, b_lr)
            lr = eta_min + 0.5 * (b_lr - eta_min) * (1.0 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs
    

if __name__=='__main__':
    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = SGD(model, 0.001)
    s=LinearDecayLR(optimizer, 100, 75)
    ss=[]
    for epoch in range(100):
        optimizer.step()
        s.step()
        ss.append(s.get_lr()[0])

    print(ss)