"""Eval OUR trained FMDetector (e.g. ab_fm_clip_ln, CLIP-L+LN trained on SBI+our-crops) on a DFDC crop
dir, with our aggregation (frame=max-face, video=mean). Tests whether our model benefits from ALIGNED
eval crops without retraining. Env: RUN (output dir), CROP_DIR, EVAL_GPU."""
import os, sys, glob, json, numpy as np, torch, pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "1")
sys.path.insert(0, os.path.join(REPO, "src"))
import cv2
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from model import FMDetector, GatedDualDetector

DFDC = os.path.join(REPO, "data/DFDC")
CROP_DIR = os.environ.get("CROP_DIR", os.path.join(REPO, "data_new/DFDC/test/frames"))
RUN = os.environ.get("RUN", "ab_fm_clip_ln")
OUT = os.path.join(REPO, "output", RUN)
cfg = json.load(open(f"{OUT}/config.json")); cfg["in_chans"] = 3
torch.backends.cuda.matmul.allow_tf32 = True
m = (GatedDualDetector(cfg) if cfg.get("model") == "gated_dual" else FMDetector(cfg)).cuda().eval()
m.load_state_dict(torch.load(f"{OUT}/weights/100.pth", map_location="cpu")["model"])

lab = pd.read_csv(f"{DFDC}/labels.csv"); lab_map = dict(zip([f.split(".")[0] for f in lab.filename], lab.label))
vids = [f.split(".")[0] for f in lab.filename]
items = []; present = np.zeros(len(vids), bool)
for vi, vid in enumerate(vids):
    cs = sorted(glob.glob(f"{CROP_DIR}/{vid}/*.png"))
    if cs: present[vi] = True
    for c in cs:
        items.append((c, vi, int(os.path.basename(c).split("_")[0].split(".")[0])))
print(f"RUN={RUN} videos={len(vids)} present={present.sum()} crops={len(items)}", flush=True)

class DS(Dataset):
    def __len__(self): return len(items)
    def __getitem__(self, i):
        p, vi, fid = items[i]
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype("float32") / 255.
        return torch.from_numpy(im.transpose(2, 0, 1)), vi, fid
dl = DataLoader(DS(), batch_size=256, num_workers=12, pin_memory=True)

fs = defaultdict(float); seen = 0
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for x, vi, fid in dl:
        p = m(x.cuda(non_blocking=True)).float().softmax(1)[:, 1].cpu().numpy()
        vi = vi.numpy(); fid = fid.numpy()
        for j in range(len(p)):
            k = (int(vi[j]), int(fid[j]))
            if p[j] > fs.get(k, -1): fs[k] = float(p[j])
        seen += len(p)
        if seen % 25600 == 0: print(f"{seen}/{len(items)}", flush=True)
vf = defaultdict(list)
for (vi, fid), s in fs.items(): vf[vi].append(s)
ys = np.array([lab_map[v] for v in vids])
ps = np.array([np.mean(vf[vi]) if vi in vf else 0.5 for vi in range(len(vids))])
print(f"\nRUN={RUN}  CROP_DIR={CROP_DIR}")
print(f"ALL n={len(ys)} (empty->0.5:{(~present).sum()})  DFDC AUC = {roc_auc_score(ys, ps):.4f}")
print(f"PRESENT n={present.sum()}  DFDC AUC = {roc_auc_score(ys[present], ps[present]):.4f}")
