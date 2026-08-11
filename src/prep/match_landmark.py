# python3 src/prep/match_landmark.py -i data/FaceForensics++/original_sequences/youtube/raw/landmarks -o data/FaceForensics++ [-v]
# 주어진 데이터셋의 모든 프레임에 대해 각각의 프레임의 landmark와 유사한 프레임을 찾아서 기록
import os
HOMEDIR = "/workspace"
os.environ["HF_HOME"] = f"{HOMEDIR}/.cache/huggingface"

import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
import json


def list_subdirs(root):
    return sorted([p for p in glob(os.path.join(root, "*")) if os.path.isdir(p)])

def list_files(folder):
    files = glob(os.path.join(folder, f"*.npy"))
    return sorted(files)


FACIAL_LANDMARKS_IDXS_81 = {
    "jaw": (0, 17), "right_eyebrow": (17, 22), "left_eyebrow": (22, 27),
    "nose": (27, 36), "right_eye": (36, 42), "left_eye": (42, 48),
    "mouth_outer": (48, 60), "mouth_inner": (60, 68), "forehead": (68, 81),
}

FACESWAP_REGION_BUDGET = {
    "right_eye":   0.28,
    "left_eye":    0.28,
    "mouth_outer": 0.24,
    "mouth_inner": 0.04,
    "nose":        0.12,
    "right_eyebrow": 0.02,
    "left_eyebrow":  0.02,
    "jaw":         0.00,
    "forehead":    0.00,
}
def make_faceswap_weights(P=81):
    w = np.zeros(P, dtype=np.float32)
    for k, budget in FACESWAP_REGION_BUDGET.items():
        i, j = FACIAL_LANDMARKS_IDXS_81[k]
        n = max(1, j - i)
        if budget > 0:
            w[i:j] = budget / n # 파트 예산을 점 개수로 균등 분배
    s = w.sum()
    return w if s == 0 else (w / s)

# 정렬 앵커(표정 누수 최소): 코끝/코기저 + 눈/입 코너
ANCHOR_IDX = np.array([30, 33, 36, 39, 42, 45, 48, 54], dtype=np.int32)
W_FACE_SWAP = make_faceswap_weights(P=81)


def find_best_matching(A, X):
    """
    GAN-based faceswap용: 앵커(코끝+눈/입 코너)로 정렬한 뒤
    '눈/입 최우선, 코 보조' per-point 가중 MSE로 스코어링.
    A: (81,2), X: (N,81,2) -> return best index
    """
    A = np.asarray(A, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    N, P = X.shape[0], X.shape[1]
    eps = 1e-12

    # ---------- 1) 정렬 앵커: 코끝 + 눈/입 코너 ----------
    A_a = A[ANCHOR_IDX]              # (Pa,2)
    X_a = X[:, ANCHOR_IDX, :]        # (N,Pa,2)

    # 중심 정렬
    A_mu = A_a.mean(axis=0, keepdims=True)        # (1,2)
    X_mu = X_a.mean(axis=1, keepdims=True)        # (N,1,2)
    A0   = A_a - A_mu                             # (Pa,2)
    X0   = X_a - X_mu                             # (N,Pa,2)

    # Kabsch 회전
    H = X0.transpose(0, 2, 1) @ A0                # (N,2,2)
    U, S, Vt = np.linalg.svd(H, full_matrices=False) # U,Vt:(N,2,2), S:(N,2)
    R = Vt.transpose(0,2,1) @ U.transpose(0,2,1) # (N,2,2)
    # no reflection(det(R)=+1)
    detR = np.linalg.det(R)
    bad = detR < 0
    if np.any(bad):
        Vt[bad, -1, :] *= -1
        R = Vt.transpose(0,2,1) @ U.transpose(0,2,1)

    # uniform scale (앵커 기준)
    denom = (X0**2).sum(axis=(1,2)).clip(min=eps) # (N,)
    s = S.sum(axis=1) / denom                      # (N,)

    # translation
    XmuR = X_mu @ R                                 # (N,1,2)
    t = (A_mu - s[:,None,None]*XmuR).reshape(-1,2)  # (N,2)

    # 전체 포인트 정렬
    X_aligned = s[:,None,None]*(X @ R) + t[:,None,:]  # (N,81,2)
    
    # ---------- 2) 가중 MSE (눈/입↑, 코 중간, 주변부↓) ----------
    diff2 = ((X_aligned - A[None,:,:])**2).sum(axis=2)   # (N,81)
    score = (diff2 * W_FACE_SWAP[None,:]).sum(axis=1) # (N,)
    j = int(np.argmin(score))
    return j


def lm_to_frame_path(npy_path: str) -> str:
    """landmarks 경로 -> frames 경로, .npy -> .png"""
    p = npy_path.replace("/landmarks/", "/frames/").replace("\\landmarks\\", "\\frames\\")
    root, _ = os.path.splitext(p)
    return root + ".png"

def make_pair_image(imgA, imgB, labelA="SRC", labelB="TRG"):
    """두 이미지를 높이 맞춰 좌우로 붙이고 라벨 텍스트를 얹어 반환"""
    if imgA is None or imgB is None:
        return None
    h1, w1 = imgA.shape[:2]; h2, w2 = imgB.shape[:2]
    target_h = min(h1, h2)
    if h1 != target_h:
        imgA = cv2.resize(imgA, (int(w1 * target_h / h1), target_h), interpolation=cv2.INTER_AREA)
    if h2 != target_h:
        imgB = cv2.resize(imgB, (int(w2 * target_h / h2), target_h), interpolation=cv2.INTER_AREA)
    pair = np.hstack([imgA, imgB])
    cv2.putText(pair, labelA, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(pair, labelB, (imgA.shape[1] + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    return pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--real_landmark_path", required=True)
    parser.add_argument("-o", "--save_path", required=True)
    parser.add_argument("-v", "--visualize", action="store_true")
    args = parser.parse_args()

    root = os.path.abspath(args.real_landmark_path)
    out_dir = os.path.abspath(args.save_path)
    os.makedirs(out_dir, exist_ok=True)

    src_vid_paths = list_subdirs(root)
    if "FaceForensics++" in root:
        list_dict = json.load(open(f'data/FaceForensics++/train.json','r'))
        filelist = []
        for i in list_dict: filelist+=i
        src_vid_paths = [i for i in src_vid_paths if os.path.basename(i)[:3] in filelist]
    trg_vid_paths = src_vid_paths
    print(f"[INFO] video-frame folders: {len(trg_vid_paths)}")

    vid_to_id = {vp:i for i, vp in enumerate(trg_vid_paths)}
    all_landmarks_list = []
    all_vid_ids = []
    all_file_paths = []

    for vp in tqdm(trg_vid_paths):
        files = list_files(vp)
        for fp in files:
            lm = np.load(fp)
            if len(lm) > 1: continue
            lm = lm[0]
            assert lm.shape == (81,2)
            all_landmarks_list.append(lm.astype(np.float32, copy=False))
            all_vid_ids.append(vid_to_id[vp])
            all_file_paths.append(fp)

    all_landmarks = np.stack(all_landmarks_list, axis=0) # (N,81,2)
    all_vid_ids = np.asarray(all_vid_ids, dtype=np.int32) # (N,)
    all_indices = np.arange(all_landmarks.shape[0], dtype=np.int64) # (N,)

    txt_path = os.path.join(out_dir, 'matched_landmarks.txt')
    vis = args.visualize
    if vis:
        vis_dir = os.path.join(out_dir, 'matched_landmarks_vis')
        os.makedirs(vis_dir, exist_ok=True)
        vis_count = 1

    with open(txt_path, 'w') as f:
        for src_vid_path in tqdm(src_vid_paths):
            src_vid_id = vid_to_id[src_vid_path]

            mask = (all_vid_ids != src_vid_id)
            trg_X = all_landmarks[mask]
            trg_idx_view = all_indices[mask] 
            
            src_file_paths = list_files(src_vid_path)
            assert len(src_file_paths) > 0
            for i, src_file_path in enumerate(src_file_paths): # for each src frame
                src_landmark = np.load(src_file_path)
                if len(src_landmark) > 1: continue # (1,81,2)
                
                j = find_best_matching(src_landmark[0], trg_X)
                best_global_idx = int(trg_idx_view[j])
                best_path = all_file_paths[best_global_idx]
                f.write(f'{src_file_path} {best_path}\n')

                if vis and i%6==5:
                    src_img_path = lm_to_frame_path(src_file_path)
                    dst_img_path = lm_to_frame_path(best_path)
                    imgA = cv2.imread(src_img_path)
                    imgB = cv2.imread(dst_img_path)
                    pair = make_pair_image(imgA, imgB, "SRC", "TRG")
                    if pair is not None:
                        out_img_path = os.path.join(vis_dir, f"pair_{vis_count}.jpg")
                        cv2.imwrite(out_img_path, pair)
                        vis_count += 1

if __name__ == "__main__":
    main()