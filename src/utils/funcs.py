import sys
import json
import numpy as np
from PIL import Image
from glob import glob 
import os
import pandas as pd
import albumentations as alb
import cv2

def load_json(path):
	d = {}
	with open(path, mode="r") as f:
		d = json.load(f)
	return d


def IoUfrom2bboxes(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou



def crop_face(img,landmark=None,bbox=None,margin=False,crop_by_bbox=True,abs_coord=False,only_img=False,phase='train'):
	assert phase in ['train','val','test']

	#crop face------------------------------------------
	H,W=len(img),len(img[0])

	assert landmark is not None or bbox is not None

	H,W=len(img),len(img[0])
	
	if crop_by_bbox:
		x0,y0=bbox[0]
		x1,y1=bbox[1]
		w=x1-x0
		h=y1-y0
		w0_margin= w/2 #w/4#0#np.random.rand()*(w/8)
		w1_margin= w/2 #w/4
		h0_margin= h/2 #h/4#0#np.random.rand()*(h/5)
		h1_margin= h/2 #h/4
	else:
		x0,y0=landmark[:68,0].min(),landmark[:68,1].min()
		x1,y1=landmark[:68,0].max(),landmark[:68,1].max()
		w=x1-x0
		h=y1-y0
		w0_margin=w/8#0#np.random.rand()*(w/8)
		w1_margin=w/8
		h0_margin=h/2#0#np.random.rand()*(h/5)
		h1_margin=h/5

	

	if margin:
		w0_margin*=4 
		w1_margin*=4 
		h0_margin*=2 
		h1_margin*=2 
	elif phase=='train':
		w0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
		w1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
		h0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
		h1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()	
	else:
		w0_margin*=0.5
		w1_margin*=0.5
		h0_margin*=0.5
		h1_margin*=0.5
			
	y0_new=max(0,int(y0-h0_margin))
	y1_new=min(H,int(y1+h1_margin)+1)
	x0_new=max(0,int(x0-w0_margin))
	x1_new=min(W,int(x1+w1_margin)+1)
	
	img_cropped=img[y0_new:y1_new,x0_new:x1_new]


	if landmark is not None:
		landmark_cropped=np.zeros_like(landmark)
		for i,(p,q) in enumerate(landmark):
			landmark_cropped[i]=[p-x0_new,q-y0_new]
	else:
		landmark_cropped=None
	if bbox is not None:
		bbox_cropped=np.zeros_like(bbox)
		for i,(p,q) in enumerate(bbox):
			bbox_cropped[i]=[p-x0_new,q-y0_new]
	else:
		bbox_cropped=None

	if only_img:
		return img_cropped
	if abs_coord:
		return img_cropped,landmark_cropped,bbox_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1),y0_new,y1_new,x0_new,x1_new
	else:
		return img_cropped,landmark_cropped,bbox_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1)


class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
	def apply(self,img,**params):
		return self.randomdownscale(img)

	def randomdownscale(self,img):
		keep_ratio=True
		keep_input_shape=True
		H,W,C=img.shape
		ratio_list=[2,4]
		r=ratio_list[np.random.randint(len(ratio_list))]
		img_ds=cv2.resize(img,(int(W/r),int(H/r)),interpolation=cv2.INTER_NEAREST)
		if keep_input_shape:
			img_ds=cv2.resize(img_ds,(W,H),interpolation=cv2.INTER_LINEAR)

		return img_ds
	

def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(alb.DualTransform):
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down,
                                          interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")

