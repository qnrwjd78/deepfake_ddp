# python3 src/prep/crop_dlib_ff_from_frame.py -dp data/FaceForensics++/manipulated_sequences/SimSwap/raw -w 4
from glob import glob
import os
import cv2
from tqdm import tqdm
import numpy as np
import argparse
import dlib
from imutils import face_utils

import threading
from concurrent.futures import ThreadPoolExecutor

_thread_local = threading.local()
def _get_models():
    if not hasattr(_thread_local, "detector"):
        _thread_local.detector = dlib.get_frontal_face_detector()
        predictor_path = 'src/prep/shape_predictor_81_face_landmarks.dat'
        _thread_local.predictor = dlib.shape_predictor(predictor_path)
    return _thread_local.detector, _thread_local.predictor


def process_one(frame_path):
    face_detector, face_predictor = _get_models()
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_detector(frame, 1)
    if len(faces)==0:
        tqdm.write(f'No detected faces for: {frame_path}')
        return

    landmarks=[]
    size_list=[]
    for face_idx in range(len(faces)):
        landmark = face_predictor(frame, faces[face_idx])
        landmark = face_utils.shape_to_np(landmark)
        x0,y0=landmark[:,0].min(),landmark[:,1].min()
        x1,y1=landmark[:,0].max(),landmark[:,1].max()
        face_s=(x1-x0)*(y1-y0)
        size_list.append(face_s)
        landmarks.append(landmark)
    landmarks=np.concatenate(landmarks).reshape((len(size_list),)+landmark.shape)
    landmarks=landmarks[np.argsort(np.array(size_list))[::-1]]

    land_path = frame_path.replace('.png','.npy').replace('/frames/','/landmarks/')
    os.makedirs(os.path.dirname(land_path), exist_ok=True)
    np.save(land_path, landmarks)
    return


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-d',dest='dataset',choices=['DeepFakeDetection_original','DeepFakeDetection','FaceShifter','Face2Face','Deepfakes','FaceSwap','NeuralTextures','Original','Celeb-real','Celeb-synthesis','YouTube-real','DFDC','DFDCP'])
    parser.add_argument('-dp',dest='dataset_path',type=str)
    parser.add_argument('-c',dest='comp',choices=['raw','c23','c40'],default='raw')
    parser.add_argument('-w',dest='max_workers',type=int,default=4)
    args=parser.parse_args()

    if args.dataset_path:
        dataset_path = args.dataset_path

    elif args.dataset=='Original':
        dataset_path='data/FaceForensics++/original_sequences/youtube/{}/'.format(args.comp)
    elif args.dataset=='DeepFakeDetection_original':
        dataset_path='data/FaceForensics++/original_sequences/actors/{}/'.format(args.comp)
    elif args.dataset in ['DeepFakeDetection','FaceShifter','Face2Face','Deepfakes','FaceSwap','NeuralTextures']:
        # dataset_path='data/FaceForensics++/manipulated_sequences/{}/{}/'.format(args.dataset,args.comp)
        dataset_path='data/FFraw/manipulated_sequences/{}/{}/'.format(args.dataset,args.comp)
    elif args.dataset in ['Celeb-real','Celeb-synthesis','YouTube-real']:
        dataset_path='data/Celeb-DF-v2/{}/'.format(args.dataset)
    elif args.dataset in ['DFDC']:
        dataset_path='data/{}/'.format(args.dataset)
    else:
        raise NotImplementedError
    
    frames_path = os.path.join(dataset_path, 'frames')
    frames_path_list = sorted(glob(os.path.join(frames_path, "**", "*.png"), recursive=True))
    print(f"{len(frames_path_list)} frames found")

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex: #
        for _ in tqdm(
            ex.map(process_one, frames_path_list),
            total=len(frames_path_list),
            desc="Processing frames"
        ):
            pass