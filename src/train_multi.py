import os
HOMEDIR = os.environ.get("HOMEDIR", "/workspace")
os.environ.setdefault("HF_HOME", f"{HOMEDIR}/.cache/huggingface")

# Cap CPU math-library threads to 1 (before numpy/torch import so OpenBLAS/MKL/OMP
# pick it up). With many DataLoader workers, each library defaulting to all cores
# spawns thousands of threads that thrash the CPU — this was the real training
# bottleneck (fixing it + cv2.setNumThreads(0) gave ~5.7x loader throughput).
# Pure threading/parallelism setting; augmentation and model outputs are unchanged.
# setdefault lets an explicit launch-time override win.
for _thr_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thr_var, "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import contextlib
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils.sbi import SBI_Multi_Dataset
from utils.scheduler import LinearDecayLR, CosineDecayLR
import argparse
import glob
from utils.logs import log


def _fmt_hms(seconds):
    """Format a duration in seconds as H:MM:SS for progress/ETA logging."""
    s = int(max(0, seconds))
    return f"{s//3600:d}:{(s%3600)//60:02d}:{s%60:02d}"
from utils.funcs import load_json
from tqdm import tqdm
from model import *
import json


def make_collate(batch_size):
    def collate(batch):
        img_fs, img_r = zip(*batch) 
        img_fs = torch.tensor(img_fs).float().transpose(0,1) # (m-1, B, c,h,w)
        img_r = torch.tensor(img_r).float().transpose(0,1) # (1, B, c,h,w)
        imgs = torch.cat([img_r, img_fs], dim=0)
        m,B = imgs.shape[:2]
        imgs = imgs.view(m*B, *imgs.shape[2:]) # (m*B, c,h,w)
        labels = torch.arange(m).repeat_interleave(repeats=B)
        if m*B == batch_size:
            return {'img': imgs, 'label': labels}

        # balanced: 각 소스(클래스 id == 0..m-1)에서 균등하게 선택
        base = batch_size // m
        rem  = batch_size - base * m
        take_per_class = [base + (1 if c < rem else 0) for c in range(m)]
        random.shuffle(take_per_class)

        sel_idx = []
        for c in range(m):
            c_idx = (labels == c).nonzero(as_tuple=False).view(-1)   # 길이 = B (각 소스는 정확히 B개)
            perm = torch.randperm(c_idx.numel())[:take_per_class[c]]
            sel_idx.append(c_idx.index_select(0, perm))
        sel_idx = torch.cat(sel_idx, dim=0)

        return {'img': imgs.index_select(0, sel_idx),
                'label': labels.index_select(0, sel_idx)}
    return collate


def make_collate_dual(batch_size):
    """dual-view (baseline_DINO): also carries 'img_wide' (global/context view). Same balanced
    selection as make_collate, applied identically to the tight and wide tensors."""
    def _stack(fs, r):
        fs = torch.tensor(fs).float().transpose(0, 1)   # (m-1, B, c,h,w)
        r  = torch.tensor(r).float().transpose(0, 1)    # (1,   B, c,h,w)
        t = torch.cat([r, fs], dim=0)                   # (m, B, ...)  -- real is class 0
        m, B = t.shape[:2]
        return t.view(m * B, *t.shape[2:]), m, B

    def collate(batch):
        img_fs, img_r, img_ws, img_rw = zip(*batch)
        imgs, m, B = _stack(img_fs, img_r)              # tight (local/FFT)
        wide, _, _ = _stack(img_ws, img_rw)             # wide  (global) -- different spatial size, own tensor
        labels = torch.arange(m).repeat_interleave(repeats=B)
        if m * B == batch_size:
            return {'img': imgs, 'img_wide': wide, 'label': labels}
        base = batch_size // m; rem = batch_size - base * m
        take_per_class = [base + (1 if c < rem else 0) for c in range(m)]
        random.shuffle(take_per_class)
        sel_idx = []
        for c in range(m):
            c_idx = (labels == c).nonzero(as_tuple=False).view(-1)
            perm = torch.randperm(c_idx.numel())[:take_per_class[c]]
            sel_idx.append(c_idx.index_select(0, perm))
        sel_idx = torch.cat(sel_idx, dim=0)
        return {'img': imgs.index_select(0, sel_idx),
                'img_wide': wide.index_select(0, sel_idx),
                'label': labels.index_select(0, sel_idx)}
    return collate

def compute_accuracy(pred, target):
    pred_idx = pred.argmax(dim=1).detach().cpu().numpy()
    true_idx = target.detach().cpu().numpy()
    return (pred_idx == true_idx).mean()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    # Single-thread the per-worker CPU libs: cv2's default thread pool + torch intra-op
    # threads otherwise oversubscribe the cores across many workers/jobs. Output-identical
    # (only disables internal op parallelism), and the dominant lever for loader throughput.
    import cv2
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(args):

    cfg=load_json(args.config)
    seed = args.seed
    cfg['seed'] = seed
    set_seed(seed)

    assert torch.cuda.is_available()
    # Multi-GPU DDP is enabled automatically when launched via torchrun (WORLD_SIZE>1);
    # a plain `python src/train_multi.py ...` run keeps the original single-GPU path.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size > 1
    if ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        local_rank, rank = 0, 0
        device = torch.device("cuda")
    is_main = (rank == 0)
    device_name = torch.cuda.get_device_name(local_rank)

    # Enable TF32 + bf16 autocast on any Ampere-or-newer GPU (sm_80+: A100, H100/H200,
    # RTX 30/40/50, ...). Was hardcoded to 'A100' only, which left H200 in slow fp32.
    major_cc = torch.cuda.get_device_capability(local_rank)[0]
    use_amp = (major_cc >= 8) and torch.cuda.is_bf16_supported()
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if is_main: print(f"AMP on: bf16 autocast + TF32 ({device_name}, sm_{major_cc}x) | DDP={ddp} world_size={world_size}")
    else:
        if is_main: print(f"AMP off: fp32 ({device_name}) | DDP={ddp} world_size={world_size}")

    image_size=cfg['image_size']
    # cfg['batch_size'] is the GLOBAL batch. Under DDP each rank processes 1/world_size of it,
    # so the effective global batch (and thus training dynamics) matches the single-GPU run.
    global_batch = cfg['batch_size']
    if ddp and global_batch % world_size != 0 and is_main:
        print(f"WARN: global batch {global_batch} not divisible by world_size {world_size}; per-GPU batch floored.")
    batch_size = global_batch // world_size if ddp else global_batch
    assert batch_size > 1, f"per-GPU batch {batch_size} must be >1 (global {global_batch}, world_size {world_size})"
    
    in_chans = 3
    cfg['in_chans'] = in_chans
    cfg['label_smoothing'] = args.label_smoothing

    train_dataset=SBI_Multi_Dataset(phase='train', image_size=image_size,
        fake_frame_paths=cfg['fake_frame_paths'], fake_shift=cfg['fake_shift'],
        real_frame_path=cfg.get('real_frame_path'), align_crop=cfg.get('align_crop', False),
        extra_sources=cfg.get('extra_train_sources'),
        dual_view=cfg.get('dual_view', False), wide_size=cfg.get('global_img_size', 224),
        degrade_strong=cfg.get('degrade_strong', False))

    # DataLoader workers: SBI augmentation is CPU-heavy and starved the GPU at the old
    # default of min(8, cpu). Scale to the box, but leave headroom so up to TRAIN_JOBS_PER_NODE
    # concurrent trainings (one per GPU) don't oversubscribe CPU. Override with NUM_WORKERS.
    cpu_total = os.cpu_count() or 8
    jobs_per_node = max(1, int(os.environ.get("TRAIN_JOBS_PER_NODE", "1")))
    # ~8 workers already saturates one GPU (99% util) once the thread-oversubscription
    # fix is in place; cap at 12 so 4 concurrent 1-per-GPU jobs stay well under 128 cores.
    default_workers = max(4, min(12, (cpu_total - 4) // jobs_per_node))
    num_workers = int(os.environ.get("NUM_WORKERS", default_workers))
    if is_main: print(f"DataLoader workers: {num_workers} (cpu={cpu_total}, jobs_per_node={jobs_per_node})")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    # sources per sample = real + SBI + every fake list (FF++ paths AND any extra_train_sources)
    m = 2 + len(train_dataset.fake_image_lists)

    # Under DDP, DistributedSampler shards the real anchors across ranks (each sees 1/world_size
    # per epoch); set_epoch() is called in the loop so the shuffle differs each epoch.
    if ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                           shuffle=True, seed=seed, drop_last=True)
    else:
        train_sampler = None

    train_loader=torch.utils.data.DataLoader(train_dataset,
        batch_size=(batch_size-1)//m+1, #
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        generator=gen,
        collate_fn=(make_collate_dual(batch_size) if cfg.get('dual_view') else make_collate(batch_size)),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker
    )
    if cfg['model'] == 'fm':
        model=FMDetector(cfg)
    elif cfg['model'] == 'gated_dual':
        model=GatedDualDetector(cfg)
    elif cfg['model'] == 'gated_dual_dino':
        from model_dino import GatedDualDINO
        model=GatedDualDINO(cfg)
    else:
        raise NotImplementedError
    
    total_params = sum(p.numel() for p in model.parameters())
    if is_main: print(f"Total params: {total_params/1e6:.2f}M")
    model=model.to(device)
    # raw_model keeps a handle to the underlying module: the SAM optimizer lives on it
    # (model.optimizer) and checkpoints save its state_dict, both of which must bypass the
    # DDP wrapper added below.
    raw_model = model

    n_epoch=cfg['epoch']
    start_epoch = 0

    cfg_sopt = cfg['scheduler']
    if cfg_sopt['name'] == "LinearDecayLR":
        lr_scheduler = LinearDecayLR(model.optimizer, n_epoch)
    elif cfg_sopt['name'] == "CosineDecayLR":
        lr_scheduler = CosineDecayLR(model.optimizer, n_epoch)

    # Output dir = <run_name>_<model> (e.g. fshntx2_gated_dual), independent of the config
    # filename, so every run folders uniformly regardless of which recipe file was used.
    save_path=f'output/{args.session_name}_{cfg["model"]}/'
    if is_main:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, "weights"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "logs"), exist_ok=True)
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(cfg, f, indent=4)
        logger = log(path=save_path+"logs/", file="losses.txt")
    else:
        logger = None
    weight_dict={}
    n_weight=2 # num saving checkpoints

    # --resume: continue an interrupted run. Checkpoints carry model/optimizer/scheduler/epoch,
    # so we restore all of them and pick up at the next epoch. '-r' alone auto-picks the
    # highest-numbered *.pth in this run's weights dir; '-r <path>' loads an explicit file.
    # Loaded on EVERY rank (before the DDP wrap) so all ranks start from identical weights and
    # each rank restores its own optimizer state.
    if args.resume:
        if args.resume == 'auto':
            ckpts = glob.glob(os.path.join(save_path, 'weights', '*.pth'))
            if not ckpts:
                raise FileNotFoundError(f"--resume auto: no checkpoints in {save_path}weights/")
            resume_path = max(ckpts, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        else:
            resume_path = args.resume
        ck = torch.load(resume_path, map_location=device)
        raw_model.load_state_dict(ck['model'])
        raw_model.optimizer.load_state_dict(ck['optimizer'])
        lr_scheduler.load_state_dict(ck['scheduler'])
        start_epoch = ck['epoch'] + 1
        msg = f"RESUME from {resume_path} -> continue at epoch {start_epoch+1}/{n_epoch}"
        if is_main: print(msg); logger.info(msg)

    # Wrap in DDP after weights are (optionally) restored: construction broadcasts rank-0
    # params to all ranks, guaranteeing a consistent start. find_unused_parameters defaults on
    # for safety (set DDP_FIND_UNUSED=0 for a small speedup once a run is confirmed healthy).
    if ddp:
        find_unused = os.environ.get("DDP_FIND_UNUSED", "1") == "1"
        model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=find_unused)

    # approximated(fixed) real:fake balanced loss weighting
    w_real = m / 2.0
    w_fake = m / (2.0 * (m-1))
    w = torch.tensor([w_real, w_fake], device=device, dtype=torch.float)

    epoch_times = []
    for epoch in range(start_epoch, n_epoch):
        epoch_t0 = time.perf_counter()
        gen.manual_seed(seed + epoch)
        if ddp:
            train_sampler.set_epoch(epoch)  # reshuffle the per-rank shard each epoch
        if cfg["fake_shuffle"]:
            assert hasattr(train_loader.dataset, "shuffle")
            train_loader.dataset.shuffle()

        model.train(mode=True)
        train_loss=0.
        train_acc=0.

        for step,data in enumerate(tqdm(train_loader, disable=not is_main)):

            img=data['img'].to(device, non_blocking=True).float()
            img_wide=data.get('img_wide')
            if img_wide is not None: img_wide=img_wide.to(device, non_blocking=True).float()
            label=data['label'].to(device, non_blocking=True).long()
            target = torch.where(label == 0, 0, 1)

            for i in range(2): # SAM 2-step, better not to handle fixed dropout
                # DDP: skip gradient all-reduce on the FIRST (ascent) pass — SAM's perturbation
                # uses local grads (the standard m-sharpness variant); only the SECOND pass, which
                # produces the actual update gradients, is synced across ranks.
                sync_ctx = model.no_sync() if (ddp and i == 0) else contextlib.nullcontext()
                with sync_ctx:
                    if use_amp:
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            pred_cls = model(img, img_wide) if img_wide is not None else model(img)
                            if i == 0: pred_first = pred_cls.detach()
                            loss = F.cross_entropy(pred_cls, target, weight=w, label_smoothing=args.label_smoothing)
                    else:
                        pred_cls = model(img, img_wide) if img_wide is not None else model(img)
                        if i == 0: pred_first = pred_cls.detach()
                        loss = F.cross_entropy(pred_cls, target, weight=w, label_smoothing=args.label_smoothing)

                    loss.backward()
                if i == 0:
                    raw_model.optimizer.first_step(zero_grad=True)
                else:
                    raw_model.optimizer.second_step(zero_grad=True)
                del pred_cls

            with torch.inference_mode(): # weighted
                output=pred_first.float()
                loss = F.cross_entropy(output, target, weight=w, label_smoothing=0.)
                train_loss+=loss.item()

                pred = output.argmax(1)
                sample_w = w.gather(0, target)
                acc_w = ((pred == target).float() * sample_w).sum() / sample_w.sum()
                train_acc += acc_w.item()
            del pred_first, output, target

        # Average the epoch's loss/acc across ranks so the logged metric matches the global
        # batch (each rank only saw its own 1/world_size shard).
        n_steps = len(train_loader)
        if ddp:
            stat = torch.tensor([train_loss, train_acc], device=device)
            dist.all_reduce(stat, op=dist.ReduceOp.SUM)
            train_loss, train_acc = stat[0].item() / world_size, stat[1].item() / world_size
        avg_loss = train_loss / n_steps
        avg_acc = train_acc / n_steps

        # Per-epoch wall time + ETA to completion (based on mean epoch time so far).
        epoch_time = time.perf_counter() - epoch_t0
        epoch_times.append(epoch_time)
        mean_epoch = sum(epoch_times) / len(epoch_times)
        remaining = n_epoch - (epoch + 1)
        eta = mean_epoch * remaining

        log_text = ("Epoch {}/{} | train w_loss: {:.4f}, train w_acc: {:.4f}, "
                    "| epoch {} ETA {} (~{}/ep)").format(
            epoch+1, n_epoch, avg_loss, avg_acc,
            _fmt_hms(epoch_time), _fmt_hms(eta), _fmt_hms(mean_epoch),
        )
        if is_main:
            logger.info(log_text)

        # Catastrophe guardrail (opt-in via env; never triggers on a healthy run).
        # Early train-loss only flags *broken* settings (divergence / not learning);
        # it does NOT rank generalization, so threshold is deliberately very loose.
        # avg_loss is already global-reduced above, so all ranks abort together.
        _cat_thr = float(os.environ.get("CATASTROPHIC_LOSS", "0"))
        _cat_after = int(os.environ.get("CATASTROPHIC_AFTER_EPOCH", "10"))
        if (not np.isfinite(avg_loss)) or (_cat_thr > 0 and (epoch + 1) >= _cat_after and avg_loss > _cat_thr):
            if is_main: logger.warning(f"CATASTROPHIC: epoch {epoch+1} avg_loss={avg_loss:.4f} (thr={_cat_thr}) -> abort")
            raise SystemExit(3)

        lr_scheduler.step()

        # save checkpoint (rank 0 only; save the raw module so the .pth loads with or without DDP)
        if is_main:
            save_model_path=os.path.join(save_path+'weights/',"{}.pth".format(epoch+1))
            state = {
                "model": raw_model.state_dict(),
                "optimizer": raw_model.optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(state, save_model_path)

            weight_dict[save_model_path] = epoch+1 #
            if len(weight_dict) > n_weight:
                worst_path = min(weight_dict, key=lambda k: weight_dict[k])
                try:
                    os.remove(worst_path)
                except Exception as e:
                    logger.warning(f"Couldn't remove old weight {worst_path}: {e}")
                del weight_dict[worst_path]

    if ddp:
        dist.barrier()          # keep ranks together until rank 0 finishes writing
        dist.destroy_process_group()
        
        
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument(dest='config')
    parser.add_argument('-n',dest='session_name',required=True)
    #parser.add_argument('-w',dest='pretrained_weight',type=str)
    parser.add_argument('-r','--resume',dest='resume',nargs='?',const='auto',default=None,
                        help="resume training: bare -r auto-picks latest checkpoint, or -r <path.pth>")
    parser.add_argument('-s',dest='seed',type=int,default=0)
    parser.add_argument('-ls',dest='label_smoothing',type=float,default=0.10)
    args=parser.parse_args()
    main(args)
