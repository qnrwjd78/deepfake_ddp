from glob import glob
import os
import json
import numpy as np
from glob import glob 
import os


def init_ff(dataset_path='data/FaceForensics++/original_sequences/youtube/raw/frames', phase='train', n_frames=8, shift=False, ff_split=True, keep_ids=None):
	if dataset_path[-1] != '/': dataset_path += '/'

	folder_list = sorted(glob(dataset_path+'*'))
	# ff_split: apply the FaceForensics++ train/val/test split (folder-id in {phase}.json).
	# Turn OFF for non-FF++ sources (e.g. in-the-wild newbench) whose folder names aren't FF++ ids.
	if ff_split:
		filelist = []
		list_dict = json.load(open(f'data/FaceForensics++/{phase}.json','r'))
		for i in list_dict:
			filelist+=i
		folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]

		# invalid idx
		invalid_idx = ['281', '604']
		folder_list = [i for i in folder_list if os.path.basename(i)[:3] not in invalid_idx]
	# keep_ids: restrict to these exact folder names (used to split a source by label)
	if keep_ids is not None:
		keep = set(keep_ids)
		folder_list = [i for i in folder_list if os.path.basename(i) in keep]
	if 'video' in dataset_path: return folder_list

	image_list=[]
	for i in range(len(folder_list)):
		images_temp=sorted(glob(folder_list[i]+'/*.png'))

		N = len(images_temp)
		if n_frames < N:
			if shift:
				if n_frames == 1:
					raise NotImplementedError
					idx = np.array([N // 2], dtype=int)
				else:
					delta = (N - 1) / (n_frames - 1)
					base = np.linspace(0, N-1, n_frames)    
					shifted = (base + 0.5 * delta) % N
					idx = (np.floor(shifted+0.5).astype(int)) % N
				images_temp = [images_temp[j] for j in idx]
			else:
				images_temp=[images_temp[round(i)] for i in np.linspace(0,N-1,n_frames)]
		image_list += images_temp
	return image_list