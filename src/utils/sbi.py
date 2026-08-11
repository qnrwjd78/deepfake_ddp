# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE

import torch
from torchvision import datasets,transforms,utils
from torch.utils.data import Dataset,IterableDataset
from glob import glob
import os
import numpy as np
from PIL import Image
import random
import cv2
import sys
import albumentations as alb
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

import logging


if os.path.isfile('./src/utils/library/bi_online_generation.py'):
	sys.path.append('./src/utils/library/')
	print('exist library')
	exist_bi=True
else:
	exist_bi=False


class SBI_Dataset(Dataset):
	def __init__(self,phase='train',image_size=224,n_frames=8,extra_modality=False):
		
		assert phase in ['train','val','test']
		
		image_list=init_ff(phase=phase,n_frames=n_frames)

		path_lm='/landmarks/' 
		image_list=[image_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
		self.path_lm=path_lm
		print(f'SBI({phase}): {len(image_list)}')
	

		self.image_list=image_list

		self.image_size=(image_size,image_size)
		self.phase=phase
		self.n_frames=n_frames

		# self.transforms=self.get_transforms()
		self.transforms=self.get_dfdc_transforms()
		self.source_transforms = self.get_source_transforms()
		self.post_transforms = self.get_post_transforms()


		# self.extra_modality = extra_modality
		# if self.extra_modality:
		# 	self.pipe = get_modality(target_modality=extra_modality)


	def __len__(self):
		return len(self.image_list)

	def __getitem__(self,idx):
		flag=True
		while flag:
			try:
				filename=self.image_list[idx]
				img=np.array(Image.open(filename))
				# cv2.imwrite(f'img_{idx}.png', img)
				landmark=np.load(filename.replace('.png','.npy').replace('/frames/',self.path_lm))[0]
				bbox_lm=np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()])
				bboxes=np.load(filename.replace('.png','.npy').replace('/frames/','/retina/'))[:2]
				iou_max=-1
				for i in range(len(bboxes)):
					iou=IoUfrom2bboxes(bbox_lm,bboxes[i].flatten())
					if iou_max<iou:
						bbox=bboxes[i]
						iou_max=iou

				landmark=self.reorder_landmark(landmark)
				if self.phase=='train':
					if np.random.rand()<0.5:
						img,_,landmark,bbox=self.hflip(img,None,landmark,bbox)
				
				img,landmark,bbox,__=crop_face(img,landmark,bbox,margin=True,crop_by_bbox=False)
				# cv2.imwrite(f'img_crop_{idx}.png', img)

				if self.phase=='train':
					img_r,img_f,mask_f=self.self_blending(img.copy(),landmark.copy())
				else:
					img_r,img_f,mask_f=self.self_blending(img.copy(),landmark.copy())
				# cv2.imwrite(f'img_blend_{idx}.png', img_f)

				if self.phase=='train':
					transformed=self.transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))
					img_f=transformed['image']
					img_r=transformed['image1']
				
				# bbox image
				# img_bbox = img.copy()
				# cv2.rectangle(img_bbox, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[1][0]), int(bbox[1][1])), (0, 255, 0), 2)

				# # landmark image
				# for i in range(len(landmark)):
				# 	# print('landmark:', landmark[i])
				# 	cv2.circle(img_bbox, (int(landmark[i][0]), int(landmark[i][1])), 2, (0, 0, 255), -1)

				bbox_landmark = np.array([[landmark[:,0].min(),landmark[:,1].min()],[landmark[:,0].max(),landmark[:,1].max()]])
				# cv2.rectangle(img_bbox, (int(bbox_landmark[0][0]), int(bbox_landmark[0][1])), 
				#   				(int(bbox_landmark[1][0]), int(bbox_landmark[1][1])), (0, 255, 0), 2)
				# cv2.imwrite(f'img_bbox_{idx}.png', img_bbox)
			

				img_f,_,__,___,y0_new,y1_new,x0_new,x1_new=crop_face(img_f,landmark,bbox_landmark,margin=False,crop_by_bbox=True,abs_coord=True,phase=self.phase)
				img_r=img_r[y0_new:y1_new,x0_new:x1_new]
				mask_f=mask_f[y0_new:y1_new,x0_new:x1_new]

				# cv2.imwrite(f'img_crop2_lm_{idx}.png', img_f)

				img_f=cv2.resize(img_f,self.image_size,interpolation=cv2.INTER_LINEAR) #.astype('float32')/255
				img_r=cv2.resize(img_r,self.image_size,interpolation=cv2.INTER_LINEAR) #.astype('float32')/255
				mask_f=cv2.resize(mask_f,self.image_size,interpolation=cv2.INTER_LINEAR)

				# save img
				# cv2.imwrite(f'img_f_{idx}.png', img_f)
				# cv2.imwrite(f'img_r_{idx}.png', img_r)
				# cv2.imwrite(f'mask_f_{idx}.png', (mask_f*255).astype(np.uint8))
				# exit()

				# cutout
				if self.phase=='train':
					length = 32
					cutout_height = random.randint(0, self.image_size[0] - length)
					cutout_width = random.randint(0, self.image_size[1] - length)
					img_f[cutout_height:cutout_height+length, cutout_width:cutout_width+length] = 0
					img_r[cutout_height:cutout_height+length, cutout_width:cutout_width+length] = 0

				
				# if self.extra_modality:
				# 	c_f, _, _ = self.pipe(img_f)
				# 	c_f = np.array(c_f).transpose((1,2,0))
				# 	c_r, _, _ = self.pipe(img_r)
				# 	c_r = np.array(c_r).transpose((1,2,0))


				img_f=img_f.astype('float32')/255
				img_r=img_r.astype('float32')/255

				# if self.extra_modality:
				# 	img_f = np.concatenate([img_f, c_f], axis=-1)
				# 	img_r = np.concatenate([img_r, c_r], axis=-1)

				img_f=img_f.transpose((2,0,1))
				img_r=img_r.transpose((2,0,1))


				flag=False
			except Exception as e:
				print(e)
				idx=torch.randint(low=0,high=len(self),size=(1,)).item()
		
		return img_f,img_r

	
		
	def get_source_transforms(self):
		return alb.Compose([
				alb.Compose([
						alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
						alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
						alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
					],p=1),
	
				alb.OneOf([
					RandomDownScale(p=1),
					alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
				],p=1),
				
			], p=1.)

	def get_dfdc_transforms(self):
		return alb.Compose([

			alb.ImageCompression(quality_lower=60,quality_upper=100,p=0.5),
			alb.GaussNoise(p=0.1),
			alb.GaussianBlur(blur_limit=3, p=0.05),
			# HorizontalFlip(),
			# alb.OneOf([
			# 	IsotropicResize(max_side=self.image_size[0], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
			# 	IsotropicResize(max_side=self.image_size[0], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
			# 	IsotropicResize(max_side=self.image_size[0], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
			# ], p=1),
			# alb.PadIfNeeded(min_height=self.image_size[0], min_width=self.image_size[1], border_mode=cv2.BORDER_CONSTANT),
			alb.OneOf([alb.RandomBrightnessContrast(), alb.FancyPCA(), alb.HueSaturationValue()], p=0.7),
			alb.ToGray(p=0.2),
			alb.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.1),
			# alb.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
		], 
		additional_targets={f'image1': 'image'},
		p=1.)

	def get_post_transforms(self):
		return alb.Compose([

			alb.OneOf([
				IsotropicResize(max_side=self.image_size[0], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
				IsotropicResize(max_side=self.image_size[0], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
				IsotropicResize(max_side=self.image_size[0], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
			], p=1),
			alb.PadIfNeeded(min_height=self.image_size[0], min_width=self.image_size[1], border_mode=cv2.BORDER_CONSTANT),
			alb.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
		], 
		additional_targets={f'image1': 'image'},
		p=1.)


	def get_transforms(self):
		return alb.Compose([
			
			alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
			alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
			alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
			alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
			alb.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 1.0), p=0.1),
			alb.ToGray(p=0.2),
			
		], 
		additional_targets={f'image1': 'image'},
		p=1.)


	def randaffine(self,img,mask):
		f=alb.Affine(
				translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
				scale=[0.95,1/0.95],
				fit_output=False,
				p=1)
			
		g=alb.ElasticTransform(
				alpha=50,
				sigma=7,
				alpha_affine=0,
				p=1,
			)

		transformed=f(image=img,mask=mask)
		img=transformed['image']
		
		mask=transformed['mask']
		transformed=g(image=img,mask=mask)
		mask=transformed['mask']
		return img,mask

		
	def self_blending(self,img,landmark):
		H,W=len(img),len(img[0])
		if np.random.rand()<0.25:
			landmark=landmark[:68]
		if exist_bi:
			logging.disable(logging.FATAL)
			mask=random_get_hull(landmark,img)[:,:,0]
			logging.disable(logging.NOTSET)
		else:
			mask=np.zeros_like(img[:,:,0])
			cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)


		source = img.copy()
		if np.random.rand()<0.5:
			source = self.source_transforms(image=source.astype(np.uint8))['image']
		else:
			img = self.source_transforms(image=img.astype(np.uint8))['image']

		source, mask = self.randaffine(source,mask)

		img_blended,mask=B.dynamic_blend(source,img,mask)
		img_blended = img_blended.astype(np.uint8)
		img = img.astype(np.uint8)

		return img,img_blended,mask
	
	def reorder_landmark(self,landmark):
		landmark_add=np.zeros((13,2))
		for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
			landmark_add[idx]=landmark[idx_l]
		landmark[68:]=landmark_add
		return landmark

	def hflip(self,img,mask=None,landmark=None,bbox=None):
		H,W=img.shape[:2]
		landmark=landmark.copy()
		bbox=bbox.copy()

		if landmark is not None:
			landmark_new=np.zeros_like(landmark)

			
			landmark_new[:17]=landmark[:17][::-1]
			landmark_new[17:27]=landmark[17:27][::-1]

			landmark_new[27:31]=landmark[27:31]
			landmark_new[31:36]=landmark[31:36][::-1]

			landmark_new[36:40]=landmark[42:46][::-1]
			landmark_new[40:42]=landmark[46:48][::-1]

			landmark_new[42:46]=landmark[36:40][::-1]
			landmark_new[46:48]=landmark[40:42][::-1]

			landmark_new[48:55]=landmark[48:55][::-1]
			landmark_new[55:60]=landmark[55:60][::-1]

			landmark_new[60:65]=landmark[60:65][::-1]
			landmark_new[65:68]=landmark[65:68][::-1]
			if len(landmark)==68:
				pass
			elif len(landmark)==81:
				landmark_new[68:81]=landmark[68:81][::-1]
			else:
				raise NotImplementedError
			landmark_new[:,0]=W-landmark_new[:,0]
			
		else:
			landmark_new=None

		if bbox is not None:
			bbox_new=np.zeros_like(bbox)
			bbox_new[0,0]=bbox[1,0]
			bbox_new[1,0]=bbox[0,0]
			bbox_new[:,0]=W-bbox_new[:,0]
			bbox_new[:,1]=bbox[:,1].copy()
			if len(bbox)>2:
				bbox_new[2,0]=W-bbox[3,0]
				bbox_new[2,1]=bbox[3,1]
				bbox_new[3,0]=W-bbox[2,0]
				bbox_new[3,1]=bbox[2,1]
				bbox_new[4,0]=W-bbox[4,0]
				bbox_new[4,1]=bbox[4,1]
				bbox_new[5,0]=W-bbox[6,0]
				bbox_new[5,1]=bbox[6,1]
				bbox_new[6,0]=W-bbox[5,0]
				bbox_new[6,1]=bbox[5,1]
		else:
			bbox_new=None

		if mask is not None:
			mask=mask[:,::-1]
		else:
			mask=None
		img=img[:,::-1].copy()
		return img,mask,landmark_new,bbox_new
	
	def collate_fn(self,batch):
		img_f,img_r=zip(*batch)
		data={}
		data['img']=torch.cat([torch.tensor(img_r).float(),torch.tensor(img_f).float()],0)
		data['label']=torch.tensor([0]*len(img_r)+[1]*len(img_f))
		return data
		
	def collate_fn_mod(self,batch):
		img_f,img_r=zip(*batch)
		data={}
		data['img_f']=img_f
		data['img_r']=img_r
		data['label']=torch.tensor([0]*len(img_r)+[1]*len(img_f))
		return data

	def worker_init_fn(self,worker_id):                                                          
		np.random.seed(np.random.get_state()[1][0] + worker_id)


class SBI_Multi_Dataset(Dataset):
	@staticmethod
	def _labels_split(labels_csv):
		"""Read a filename,label CSV -> (real_ids, fake_ids) sets of video ids (basename w/o ext)."""
		import csv
		real, fake = set(), set()
		with open(labels_csv) as f:
			for row in csv.DictReader(f):
				vid = row['filename'].rsplit('.', 1)[0]
				(real if int(row['label']) == 0 else fake).add(vid)
		return real, fake

	def __init__(self, phase='train', image_size=224, n_frames=8, fake_frame_paths=[], fake_shift=False, real_frame_path=None, align_crop=False, extra_sources=None, dual_view=False, wide_size=224, degrade_strong=False):
		if phase != 'train': raise NotImplementedError
		# align_crop: emit DeepfakeBench-style 5-pt similarity-aligned crops (scale 1.3) instead of the
		# loose landmark-bbox crop, so training crops MATCH the aligned eval crops (the +0.033 lever).
		self.align_crop = align_crop
		# dual_view (baseline_DINO): also emit the WIDE view (the margin/loose crop, pre-tight-crop),
		# resized to wide_size, so a global backbone (DINOv3) can see context while local sees the tight crop.
		self.dual_view = dual_view
		self.wide_size = int(wide_size)
		_res = image_size; _dst = np.array([[30.2946,51.6963],[65.5318,51.5014],[48.0252,71.7366],
			[33.5493,92.3655],[62.7299,92.2041]], np.float32)
		_dst[:,0]+=8.0; _dst[:,0]*=_res/112.; _dst[:,1]*=_res/112.
		_mr=0.3; _xm=_res*_mr/2.; _ym=_res*_mr/2.
		_dst[:,0]+=_xm; _dst[:,1]+=_ym; _dst[:,0]*=_res/(_res+2*_xm); _dst[:,1]*=_res/(_res+2*_ym)
		self._dst5 = _dst

		# real_frame_path lets us swap the SBI real source (e.g. c23 instead of raw)
		# to isolate the compression variable. Defaults to the raw youtube frames.
		if real_frame_path:
			real_image_list = init_ff(dataset_path=real_frame_path, phase=phase, n_frames=n_frames)
		else:
			real_image_list = init_ff(phase=phase, n_frames=n_frames)
		path_lm='/landmarks/' 
		real_image_list=[real_image_list[i] for i in range(len(real_image_list)) if os.path.isfile(real_image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(real_image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
		self.path_lm=path_lm
		def _lmret(lst):  # keep frames that have BOTH a landmark and a retina .npy
			return [p for p in lst
					if os.path.isfile(p.replace('/frames/',path_lm).replace('.png','.npy'))
					and os.path.isfile(p.replace('/frames/','/retina/').replace('.png','.npy'))]
		# extra_sources: non-FF++ datasets (e.g. in-the-wild newbench) as [{"frames":..,"labels":csv}].
		# label-0 videos join the real/SBI pool; label-1 videos become extra fake source lists.
		# "fake_only": true  -> use ONLY this source's fakes (skip its reals), e.g. to add newbench2
		# fakes without growing the real pool (which is what drives training time).
		_extra_fakes = []
		for _src in (extra_sources or []):
			_rids, _fids = self._labels_split(_src['labels'])
			if _src.get('fake_only'):
				_er = []
			else:
				_er = _lmret(init_ff(dataset_path=_src['frames'], phase=phase, n_frames=n_frames, ff_split=False, keep_ids=_rids))
				real_image_list += _er
			_ef = _lmret(init_ff(dataset_path=_src['frames'], n_frames=n_frames, shift=fake_shift, ff_split=False, keep_ids=_fids))
			_extra_fakes.append((_src['frames'], _ef))
			print(f"[extra] {_src['frames']}: real+{len(_er)} fake+{len(_ef)}{' (fake_only)' if _src.get('fake_only') else ''}")

		print(f'Real: {len(real_image_list)}', end=' | ')
		print(f'SBI Fake: {len(real_image_list)}', end=' | ')	

		fake_image_lists = []
		for i, fake_path in enumerate(fake_frame_paths):
			fake_image_list = init_ff(dataset_path=fake_path, n_frames=n_frames, shift=fake_shift)
			fake_image_list = [fake_image_list[i] for i in range(len(fake_image_list)) if os.path.isfile(fake_image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(fake_image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
			if len(real_image_list) < len(fake_image_list):
				fake_image_list = random.sample(fake_image_list, len(real_image_list))
			
			assert len(fake_image_list) > 0
			fake_image_lists.append(fake_image_list)
			print(f'Added_Fake{i}: {len(fake_image_list)}', end=' | ')
		# append extra fake source lists (label-1 videos from extra_sources)
		for _j, (_frames, _ef) in enumerate(_extra_fakes):
			if len(real_image_list) < len(_ef):
				_ef = random.sample(_ef, len(real_image_list))
			assert len(_ef) > 0, f"no usable fake frames in extra source {_frames}"
			fake_image_lists.append(_ef)
			print(f'Added_ExtraFake{_j}: {len(_ef)}', end=' | ')
		print()

		self.phase = phase
		self.real_image_list=real_image_list
		self.fake_image_lists=fake_image_lists
		self.image_size=(image_size,image_size)
		self.transforms=self.get_dfdc_transforms_strong() if degrade_strong else self.get_dfdc_transforms()
		self.source_transforms = self.get_source_transforms()

	def __len__(self):
		return len(self.real_image_list)

	def shuffle(self):
		#random.shuffle(self.real_image_list)
		for lst in self.fake_image_lists:
			random.shuffle(lst)

	def __getitem__(self,idx):
		img_fs = []; img_ws = []
		while True:
			try:
				if self.dual_view:
					img_r, img_f, img_rw, img_fw = self.get_fake(self.real_image_list[idx], sbi=True)
				else:
					img_r, img_f = self.get_fake(self.real_image_list[idx], sbi=True)
				break
			except Exception as e:
				print(e)
				idx=torch.randint(low=0,high=len(self),size=(1,)).item()
		img_fs.append(img_f)
		if self.dual_view: img_ws.append(img_fw)

		for i in range(len(self.fake_image_lists)):
			src = self.fake_image_lists[i][idx % len(self.fake_image_lists[i])]
			if self.dual_view:
				img_f, img_fw = self.get_fake(src, sbi=False)
				img_fs.append(img_f); img_ws.append(img_fw)
			else:
				img_fs.append(self.get_fake(src, sbi=False))
		if self.dual_view:
			return img_fs, [img_r], img_ws, [img_rw]
		return img_fs, [img_r]

	def _align_M(self, landmark):
		# 5-pt (eyes/nose/mouth corners) similarity transform to the canonical template
		src = landmark[[37,44,30,49,55], :2].astype(np.float32)
		M, _ = cv2.estimateAffinePartial2D(src, self._dst5, method=cv2.LMEDS)
		if M is None:
			M = np.array([[1.,0.,0.],[0.,1.,0.]], np.float32)
		return M

	def get_fake(self, filename, sbi=True):
		
		img=np.array(Image.open(filename))
		landmark=np.load(filename.replace('.png','.npy').replace('/frames/',self.path_lm))[0]
		bbox_lm=np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()])
		bboxes=np.load(filename.replace('.png','.npy').replace('/frames/','/retina/'))[:2]
		iou_max=-1
		for i in range(len(bboxes)):
			iou=IoUfrom2bboxes(bbox_lm,bboxes[i].flatten())
			if iou_max<iou:
				bbox=bboxes[i]
				iou_max=iou

		landmark=self.reorder_landmark(landmark)
		if np.random.rand()<0.5:
			img,_,landmark,bbox=self.hflip(img,None,landmark,bbox)
		img,landmark,bbox,__=crop_face(img,landmark,bbox,margin=True,crop_by_bbox=False)

		if sbi: 
			img_r,img_f,_=self.self_blending(img.copy(),landmark.copy())
			transformed=self.transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))
			img_f=transformed['image']
			img_r=transformed['image1']
			if self.dual_view:  # WIDE = margin crop (pre-tight-crop), resized to wide_size
				wr=cv2.resize(img_r,(self.wide_size,self.wide_size),interpolation=cv2.INTER_LINEAR).astype('float32')/255
				wf=cv2.resize(img_f,(self.wide_size,self.wide_size),interpolation=cv2.INTER_LINEAR).astype('float32')/255
				wr=wr.transpose((2,0,1)); wf=wf.transpose((2,0,1))
			if self.align_crop:
				M=self._align_M(landmark)
				img_f=cv2.warpAffine(img_f,M,self.image_size,flags=cv2.INTER_LINEAR)
				img_r=cv2.warpAffine(img_r,M,self.image_size,flags=cv2.INTER_LINEAR)
			else:
				bbox_landmark = np.array([[landmark[:,0].min(),landmark[:,1].min()],[landmark[:,0].max(),landmark[:,1].max()]])
				img_f,_,__,___,y0_new,y1_new,x0_new,x1_new=crop_face(img_f,landmark,bbox_landmark,margin=False,crop_by_bbox=True,abs_coord=True,phase=self.phase)
				img_r=img_r[y0_new:y1_new,x0_new:x1_new]
				img_f=cv2.resize(img_f,self.image_size,interpolation=cv2.INTER_LINEAR) #.astype('float32')/255
				img_r=cv2.resize(img_r,self.image_size,interpolation=cv2.INTER_LINEAR) #.astype('float32')/255

			# cutout
			length = 32
			cutout_height = random.randint(0, self.image_size[0] - length)
			cutout_width = random.randint(0, self.image_size[1] - length)
			img_f[cutout_height:cutout_height+length, cutout_width:cutout_width+length] = 0
			img_r[cutout_height:cutout_height+length, cutout_width:cutout_width+length] = 0

			img_f=img_f.astype('float32')/255
			img_r=img_r.astype('float32')/255

			img_f=img_f.transpose((2,0,1))
			img_r=img_r.transpose((2,0,1))
			if self.dual_view:
				return img_r, img_f, wr, wf
			return img_r, img_f # np array, (3,img_size,img_size), range [0,1]

		else:
			transformed = self.transforms(image=img.astype('uint8'))
			img_f = transformed['image']
			if self.dual_view:  # WIDE = margin crop, resized to wide_size
				wf=cv2.resize(img_f,(self.wide_size,self.wide_size),interpolation=cv2.INTER_LINEAR).astype('float32')/255
				wf=wf.transpose((2,0,1))
			if self.align_crop:
				img_f = cv2.warpAffine(img_f, self._align_M(landmark), self.image_size, flags=cv2.INTER_LINEAR)
			else:
				bbox_landmark = np.array([[landmark[:,0].min(),landmark[:,1].min()],[landmark[:,0].max(),landmark[:,1].max()]])
				img_f, _, __, ___, _, _, _, _ = crop_face(
	                img_f, landmark, bbox_landmark,
	                margin=False, crop_by_bbox=True, abs_coord=True, phase=self.phase
	            )
				img_f = cv2.resize(img_f, self.image_size, interpolation=cv2.INTER_LINEAR)
	
			# cutout
			length = 32
			cutout_height = random.randint(0, self.image_size[0] - length)
			cutout_width  = random.randint(0, self.image_size[1] - length)
			img_f[cutout_height:cutout_height+length, cutout_width:cutout_width+length] = 0

			img_f = img_f.astype('float32') / 255.0
			img_f = img_f.transpose((2,0,1))
			if self.dual_view:
				return img_f, wf
			return img_f  # shape: (3, H, W), range: [0,1]

	def get_dfdc_transforms_strong(self):
		# Heavy DEGRADATION to close the train/eval gap: eval2024 real crops are ~3x less sharp
		# (lapVar ~30 vs ~110) and lower-res (up-scaled social-media re-encodes). The mild version
		# has NO downscale, so add it as the key lever. Applied to BOTH real & fake (additional_targets,
		# same random params) so degradation is NOT a real/fake cue. p=0.5 downscale keeps half the
		# samples sharp -> preserves the subtle SBI blend boundary and DFDC (already sharp-ish).
		# Calibrated (scratchpad): downscale 0.35-0.7 + q40-95 spans lapVar ~40-70 (DFDC<->eval2024).
		return alb.Compose([
			alb.Downscale(scale_min=0.3, scale_max=0.6,
			              interpolation=dict(downscale=cv2.INTER_AREA, upscale=cv2.INTER_LINEAR), p=0.7),
			alb.ImageCompression(quality_lower=35, quality_upper=90, p=0.7),
			alb.GaussNoise(p=0.1),
			alb.GaussianBlur(blur_limit=3, p=0.1),
			alb.OneOf([alb.RandomBrightnessContrast(), alb.FancyPCA(), alb.HueSaturationValue()], p=0.7),
			alb.ToGray(p=0.2),
			alb.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.1),
		],
		additional_targets={f'image1': 'image'},
		p=1.)

	def get_source_transforms(self):
		return alb.Compose([
				alb.Compose([
						alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
						alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
						alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
					],p=1),
	
				alb.OneOf([
					RandomDownScale(p=1),
					alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
				],p=1),
				
			], p=1.)


	def get_dfdc_transforms(self):
		return alb.Compose([

			alb.ImageCompression(quality_lower=60,quality_upper=100,p=0.5),
			alb.GaussNoise(p=0.1),
			alb.GaussianBlur(blur_limit=3, p=0.05),
			# HorizontalFlip(),
			# alb.OneOf([
			# 	IsotropicResize(max_side=self.image_size[0], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
			# 	IsotropicResize(max_side=self.image_size[0], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
			# 	IsotropicResize(max_side=self.image_size[0], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
			# ], p=1),
			# alb.PadIfNeeded(min_height=self.image_size[0], min_width=self.image_size[1], border_mode=cv2.BORDER_CONSTANT),
			alb.OneOf([alb.RandomBrightnessContrast(), alb.FancyPCA(), alb.HueSaturationValue()], p=0.7),
			alb.ToGray(p=0.2),
			alb.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.1),
			# alb.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
		], 
		additional_targets={f'image1': 'image'},
		p=1.)

	def get_post_transforms(self):
		return alb.Compose([

			alb.OneOf([
				IsotropicResize(max_side=self.image_size[0], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
				IsotropicResize(max_side=self.image_size[0], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
				IsotropicResize(max_side=self.image_size[0], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
			], p=1),
			alb.PadIfNeeded(min_height=self.image_size[0], min_width=self.image_size[1], border_mode=cv2.BORDER_CONSTANT),
			alb.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
		], 
		additional_targets={f'image1': 'image'},
		p=1.)


	def get_transforms(self):
		return alb.Compose([
			alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
			alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
			alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
			alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
			alb.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 1.0), p=0.1),
			alb.ToGray(p=0.2),
			
		], 
		additional_targets={f'image1': 'image'},
		p=1.)


	def randaffine(self,img,mask):
		f=alb.Affine(
				translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
				scale=[0.95,1/0.95],
				fit_output=False,
				p=1)
			
		g=alb.ElasticTransform(
				alpha=50,
				sigma=7,
				alpha_affine=0,
				p=1,
			)

		transformed=f(image=img,mask=mask)
		img=transformed['image']
		
		mask=transformed['mask']
		transformed=g(image=img,mask=mask)
		mask=transformed['mask']
		return img,mask

		
	def self_blending(self,img,landmark):
		H,W=len(img),len(img[0])
		if np.random.rand()<0.25:
			landmark=landmark[:68]
		if exist_bi:
			logging.disable(logging.FATAL)
			mask=random_get_hull(landmark,img)[:,:,0]
			logging.disable(logging.NOTSET)
		else:
			mask=np.zeros_like(img[:,:,0])
			cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)


		source = img.copy()
		if np.random.rand()<0.5:
			source = self.source_transforms(image=source.astype(np.uint8))['image']
		else:
			img = self.source_transforms(image=img.astype(np.uint8))['image']

		source, mask = self.randaffine(source,mask)

		img_blended,mask=B.dynamic_blend(source,img,mask)
		img_blended = img_blended.astype(np.uint8)
		img = img.astype(np.uint8)

		return img,img_blended,mask
	
	def reorder_landmark(self,landmark):
		landmark_add=np.zeros((13,2))
		for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
			landmark_add[idx]=landmark[idx_l]
		landmark[68:]=landmark_add
		return landmark

	def hflip(self,img,mask=None,landmark=None,bbox=None):
		H,W=img.shape[:2]
		landmark=landmark.copy()
		bbox=bbox.copy()

		if landmark is not None:
			landmark_new=np.zeros_like(landmark)

			
			landmark_new[:17]=landmark[:17][::-1]
			landmark_new[17:27]=landmark[17:27][::-1]

			landmark_new[27:31]=landmark[27:31]
			landmark_new[31:36]=landmark[31:36][::-1]

			landmark_new[36:40]=landmark[42:46][::-1]
			landmark_new[40:42]=landmark[46:48][::-1]

			landmark_new[42:46]=landmark[36:40][::-1]
			landmark_new[46:48]=landmark[40:42][::-1]

			landmark_new[48:55]=landmark[48:55][::-1]
			landmark_new[55:60]=landmark[55:60][::-1]

			landmark_new[60:65]=landmark[60:65][::-1]
			landmark_new[65:68]=landmark[65:68][::-1]
			if len(landmark)==68:
				pass
			elif len(landmark)==81:
				landmark_new[68:81]=landmark[68:81][::-1]
			else:
				raise NotImplementedError
			landmark_new[:,0]=W-landmark_new[:,0]
			
		else:
			landmark_new=None

		if bbox is not None:
			bbox_new=np.zeros_like(bbox)
			bbox_new[0,0]=bbox[1,0]
			bbox_new[1,0]=bbox[0,0]
			bbox_new[:,0]=W-bbox_new[:,0]
			bbox_new[:,1]=bbox[:,1].copy()
			if len(bbox)>2:
				bbox_new[2,0]=W-bbox[3,0]
				bbox_new[2,1]=bbox[3,1]
				bbox_new[3,0]=W-bbox[2,0]
				bbox_new[3,1]=bbox[2,1]
				bbox_new[4,0]=W-bbox[4,0]
				bbox_new[4,1]=bbox[4,1]
				bbox_new[5,0]=W-bbox[6,0]
				bbox_new[5,1]=bbox[6,1]
				bbox_new[6,0]=W-bbox[5,0]
				bbox_new[6,1]=bbox[5,1]
		else:
			bbox_new=None

		if mask is not None:
			mask=mask[:,::-1]
		else:
			mask=None
		img=img[:,::-1].copy()
		return img,mask,landmark_new,bbox_new


	def worker_init_fn(self,worker_id):                                                          
		np.random.seed(np.random.get_state()[1][0] + worker_id)




if __name__=='__main__':
	import blend as B
	from initialize import *
	from funcs import IoUfrom2bboxes,crop_face,RandomDownScale
	if exist_bi:
		from library.bi_online_generation import random_get_hull
	seed=10
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	image_dataset=SBI_Dataset(phase='test',image_size=256)
	batch_size=64
	dataloader = torch.utils.data.DataLoader(image_dataset,
					batch_size=batch_size,
					shuffle=True,
					collate_fn=image_dataset.collate_fn,
					num_workers=0,
					worker_init_fn=image_dataset.worker_init_fn
					)
	data_iter=iter(dataloader)
	data=next(data_iter)
	img=data['img']
	img=img.view((-1,3,256,256))
	utils.save_image(img, 'loader.png', nrow=batch_size, normalize=False, range=(0, 1))
else:
	from utils import blend as B
	from .initialize import *
	from .funcs import IoUfrom2bboxes,crop_face,RandomDownScale,IsotropicResize
	if exist_bi:
		from utils.library.bi_online_generation import random_get_hull