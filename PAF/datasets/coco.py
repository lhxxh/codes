import copy
import json
import math
import os
import pickle

import cv2
import numpy as np
import pycocotools
import pycocotools.coco as coco
import albumentations as A

from torch.utils.data.dataset import Dataset
import torch
import numpy as np
import sys

'''
BODY_PARTS_KPT_IDS = [[1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 2], [2, 3], [3, 4], [2, 16],
                      [1, 5], [5, 6], [6, 7], [5, 17], [1, 0], [0, 14], [0, 15], [14, 16], [15, 17]]


def get_mask(segmentations, mask):
    for segmentation in segmentations:
        rle = pycocotools.mask.frPyObjects(segmentation, mask.shape[0], mask.shape[1])
        mask[pycocotools.mask.decode(rle) > 0.5] = 0
    return mask
'''

class CocoTrainDataset(Dataset):
    def __init__(self, images_folder, stride, sigma, paf_thickness, transform=None):
        super().__init__()
        self._images_folder = images_folder
        self._stride = stride
        self._sigma = sigma
        self._paf_thickness = paf_thickness
        '''
        self._transform = transform
        with open(labels, 'rb') as f:
            self._labels = pickle.load(f)
        '''
        '''    
        image = cv2.imread(os.path.join(self._images_folder, '000000390348.jpg'), cv2.IMREAD_COLOR)
        cv2.imwrite('pictures/{}.jpg'.format(torch.cuda.current_device()),image)
        '''
        ###########################################################################################
        
        self._data_dir = '/home/hl3424@columbia.edu/PAF/openpose_pytorch/COCO/annotations/'
        self._annot_path =  os.path.join(self._data_dir,'instances_train2017.json')    
        self._max_objs = 128
        
        self._class_name = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self._valid_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
        24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
        37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
        58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
        82, 84, 85, 86, 87, 88, 89, 90]
        self._cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        
        print('==> initializing coco 2017 training data.')
        self._coco = coco.COCO(self._annot_path)
        self._images = self._coco.getImgIds()
        self._num_samples = len(self._images)

        print('Loaded training {} samples'.format(self._num_samples))
        
        #############################################################################################
        '''
        _labels {'img_paths': '000000390348.jpg', 'img_width': 640, 'img_height': 428, 'objpos': [203.14000000000001, 137.46], 'image_id': 390348, 'bbox': [167.11, 74.51,
        72.06, 125.9], 'segment_area': 3463.4542, 'scale_provided': 0.34211956521739134, 'num_keypoints': 11, 'segmentations': [], 'keypoints': [[0, 0, 2], [0, 0, 2], [0, 0, 2], [193, 88, 1
        ], [211, 89, 1], [179, 110, 1], [215, 108, 1], [179, 137, 0], [225, 135, 1], [0, 0, 2], [232, 155, 1], [189, 163, 1], [218, 162, 1], [189, 203, 0], [218, 202, 0], [0, 0, 2], [0, 0, 2
        ]], 'processed_other_annotations': [{'objpos': [73.88, 149.875], 'bbox': [48.51, 56.64, 50.74, 186.47], 'segment_area': 5347.15085, 'scale_provided': 0.5067119565217392, 'num_keypoin
        ts': 12, 'keypoints': [[0, 0, 2], [0, 0, 2], [0, 0, 2], [62, 72, 1], [79, 71, 1], [57, 94, 1], [91, 90, 1], [53, 119, 1], [92, 113, 1], [0, 0, 2], [0, 0, 2], [65, 147, 1], [88, 146, 
        0], [64, 190, 1], [87, 190, 1], [67, 234, 1], [90, 231, 1]]}, {'objpos': [29.275, 149.87], 'bbox': [1.24, 58.05, 56.07, 183.64], 'segment_area': 6208.85905, 'scale_provided': 0.49902
        173913043474, 'num_keypoints': 11, 'keypoints': [[0, 0, 2], [0, 0, 2], [0, 0, 2], [21, 79, 1], [41, 79, 1], [13, 100, 1], [50, 98, 1], [8, 124, 1], [0, 0, 2], [0, 0, 2], [0, 0, 2], [
        22, 149, 1], [47, 149, 1], [24, 192, 1], [48, 194, 1], [27, 231, 1], [49, 231, 1]]}, {'objpos': [474.16999999999996, 101.47], 'bbox': [452.39, 49, 43.56, 104.94], 'segment_area': 275
        4.2988, 'scale_provided': 0.2851630434782609, 'num_keypoints': 14, 'keypoints': [[0, 0, 2], [0, 0, 2], [0, 0, 2], [470, 58, 1], [482, 58, 0], [466, 68, 1], [483, 68, 1], [461, 87, 1]
        , [490, 84, 1], [460, 101, 0], [490, 98, 1], [471, 95, 1], [480, 95, 1], [470, 120, 1], [480, 118, 1], [472, 141, 1], [476, 141, 1]]}]}
        
        bbox (x,y,width,height) x+width/y+height x:from image left to right / y:from top to bottom
        processed_other_annotations contain objects in case there are more than one objects in an image
        several same img_paths exists to handle more than one object case
        '''

    def __getitem__(self, idx):
        '''
        label = copy.deepcopy(self._labels[idx])  # label modified in transform
        image = cv2.imread(os.path.join(self._images_folder, label['img_paths']), cv2.IMREAD_COLOR)
        mask = np.ones(shape=(label['img_height'], label['img_width']), dtype=np.float32)
        mask = get_mask(label['segmentations'], mask)
        sample = {
            'label': label,
            'image': image,
            'mask': mask
        }
        if self._transform:
            sample = self._transform(sample)

        mask = cv2.resize(sample['mask'], dsize=None, fx=1/self._stride, fy=1/self._stride, interpolation=cv2.INTER_AREA)
        keypoint_maps = self._generate_keypoint_maps(sample)
        sample['keypoint_maps'] = keypoint_maps
        keypoint_mask = np.zeros(shape=keypoint_maps.shape, dtype=np.float32)
        for idx in range(keypoint_mask.shape[0]):
            keypoint_mask[idx] = mask
        sample['keypoint_mask'] = keypoint_mask

        paf_maps = self._generate_paf_maps(sample)
        sample['paf_maps'] = paf_maps
        paf_mask = np.zeros(shape=paf_maps.shape, dtype=np.float32)
        for idx in range(paf_mask.shape[0]):
            paf_mask[idx] = mask
        sample['paf_mask'] = paf_mask

        image = sample['image'].astype(np.float32)          # image size (368,368) 
        image = (image - 128) / 256                         # pixel value range from -0.5 to 0.5
        sample['image'] = image.transpose((2, 0, 1))
        del sample['label']                                 # sample dict_keys(['image', 'mask', 'keypoint_maps', 'keypoint_mask', 'paf_maps', 'paf_mask'])
        return sample
        '''
        #############################################################################################################################

        img_id = self._images[idx]
        file_name = self._coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self._images_folder, file_name)
        ann_ids = self._coco.getAnnIds(imgIds=[img_id])
        anns = self._coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self._max_objs)   
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        height, width = img.shape[0], img.shape[1]
        
        transform = A.Compose([
        A.ShiftScaleRotate(),
        A.RandomSizedBBoxSafeCrop(height=368, width=368),
        A.HorizontalFlip(p=0.5)
        ], bbox_params=A.BboxParams(format='coco', min_area=16, min_visibility=0.05,label_fields=['class_labels']))  
        
        bboxes = []
        class_labels = []
        
        for ann in anns:
            if (ann['bbox'][2] >= 1) and (ann['bbox'][3] >= 1):  # handle xmax = xmin || ymax = ymin case
                bboxes.append(ann['bbox'])
                class_labels.append(int(self._cat_ids[ann['category_id']])+1)  # background is 0, person 1, ....    
        
        transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_class_labels = transformed['class_labels']
        
        '''
        random = np.random.randint(1000)
        with open("pictures/{}_before.txt".format(random),"w") as f1:
            f1.write('before transformation : '+ np.array2string(np.asarray(bboxes))+' class labels '+ np.array2string(np.asarray(class_labels))) 
        cv2.imwrite('pictures/{}_before.jpg'.format(random),img)
        with open("pictures/{}_after.txt".format(random),"w") as f2:
            f2.write('after transformation : '+ np.array2string(np.asarray(transformed_bboxes))+' class labels '+ np.array2string(np.asarray(transformed_class_labels)))        
        cv2.imwrite('pictures/{}_after.jpg'.format(random),transformed_image)
        '''
        
        label = []
        for idx,coco_box in enumerate(transformed_bboxes):
            bbox_center = self._coco_box_center(coco_box)
            bbox = self._coco_box_to_bbox(coco_box)
            bbox_ann = {}
            bbox_ann['objpos'] = bbox_center
            bbox_ann['bbox'] = bbox
            #bbox_ann['class_label'] = transformed_class_labels[idx]
            label.append(bbox_ann)
            
        image = transformed_image
        sample = {
            'label': label,
            'image': image,
        }      

        ################################ above dataloading and data augmentation are correct ###################################
        keypoint_maps = self._generate_keypoint_maps(sample)
        sample['keypoint_maps'] = keypoint_maps        
        paf_maps = self._generate_paf_maps(sample)
        sample['paf_maps'] = paf_maps
        '''
        random = np.random.randint(1000)
        cv2.imwrite('hm_paf/{}_image.jpg'.format(random),sample['image'])
        np.set_printoptions(threshold=sys.maxsize)
        for ii,bbox_ann in enumerate(sample['label']):
            class_label = bbox_ann['class_label']
            
            tl = cv2.normalize(sample['keypoint_maps'][class_label],None,0,255,cv2.NORM_MINMAX)
            tl = cv2.applyColorMap(tl.astype(np.uint8),cv2.COLORMAP_JET)
            cv2.imwrite('hm_paf/{}_{}_tl.jpg'.format(ii,random),tl)
            md = cv2.normalize(sample['keypoint_maps'][class_label+80],None,0,255,cv2.NORM_MINMAX)
            md = cv2.applyColorMap(md.astype(np.uint8),cv2.COLORMAP_JET)
            cv2.imwrite('hm_paf/{}_{}_md.jpg'.format(ii,random),md)
            br = cv2.normalize(sample['keypoint_maps'][class_label+160],None,0,255,cv2.NORM_MINMAX)
            br = cv2.applyColorMap(br.astype(np.uint8),cv2.COLORMAP_JET)
            cv2.imwrite('hm_paf/{}_{}_br.jpg'.format(ii,random),br)
            with open("hm_paf/{}_{}_class.txt".format(ii,random),"w") as f1:
                f1.write('current class is '+str(class_label)) 
            
              
            with open("hm_paf/{}_{}_md_tl_x.txt".format(ii,random),"w") as f2:
                f2.write(np.array2string(sample['paf_maps'][2*(class_label-1)]))
            with open("hm_paf/{}_{}_md_tl_y.txt".format(ii,random),"w") as f3:
                f3.write(np.array2string(sample['paf_maps'][2*(class_label-1)+1]))
            with open("hm_paf/{}_{}_md_br_x.txt".format(ii,random),"w") as f4:
                f4.write(np.array2string(sample['paf_maps'][2*(class_label-1)+160]))
            with open("hm_paf/{}_{}_md_br_y.txt".format(ii,random),"w") as f5:
                f5.write(np.array2string(sample['paf_maps'][2*(class_label-1)+161]))
            
            
            md_tl_x = sample['paf_maps'][2*(class_label-1)]
            if np.all(md_tl_x >= 0):
                print('errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')
            md_tl_y = sample['paf_maps'][2*(class_label-1)+1]
            if np.all(md_tl_y >= 0):
                print('errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')     
            md_br_x = sample['paf_maps'][2*(class_label-1)+160]
            if np.all(md_br_x <= 0):
                print('errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')
            md_br_y = sample['paf_maps'][2*(class_label-1)+161]
            if np.all(md_br_y <= 0):
                print('errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')      
                
            md_tl_x = cv2.normalize(np.abs(md_tl_x),None,0,255,cv2.NORM_MINMAX)
            md_tl_x = cv2.applyColorMap(md_tl_x.astype(np.uint8),cv2.COLORMAP_JET)
            cv2.imwrite('hm_paf/{}_{}_md_tl_x.jpg'.format(ii,random),md_tl_x)
            md_tl_y = cv2.normalize(np.abs(md_tl_y),None,0,255,cv2.NORM_MINMAX)
            md_tl_y = cv2.applyColorMap(md_tl_y.astype(np.uint8),cv2.COLORMAP_JET)  
            cv2.imwrite('hm_paf/{}_{}_md_tl_y.jpg'.format(ii,random),md_tl_y)
            md_br_x = cv2.normalize(np.abs(md_br_x),None,0,255,cv2.NORM_MINMAX)
            md_br_x = cv2.applyColorMap(md_br_x.astype(np.uint8),cv2.COLORMAP_JET)
            cv2.imwrite('hm_paf/{}_{}_md_br_x.jpg'.format(ii,random),md_br_x)
            md_br_y = cv2.normalize(np.abs(md_br_y),None,0,255,cv2.NORM_MINMAX)
            md_br_y = cv2.applyColorMap(md_br_y.astype(np.uint8),cv2.COLORMAP_JET)              
            cv2.imwrite('hm_paf/{}_{}_md_br_y.jpg'.format(ii,random),md_br_y)
        '''    
        
        ################################ above heatmap and paf are correct ###########################################  
        image = sample['image'].astype(np.float32)          # image size (368,368) 
        image = (image - 128) / 256                         # pixel value range from -0.5 to 0.5
        sample['image'] = image.transpose((2, 0, 1))
        del sample['label']                                 # sample dict_keys(['image','keypoint_maps','paf_maps'])
        return sample        
        ################################################################################################################################
        
        '''
        anns example : [{'segmentation': [[195.8, 627.79, 221.83, 533.29, 182.11, 485.34, 208.12, 371.65, 220.46, 333.3, 209.5, 270.3, 201.28, 256.59, 177.99, 242.9, 151.97, 
        255.22, 135.54, 278.51, 117.73, 256.59, 117.73, 234.69, 141.02, 238.78, 173.88, 212.77, 191.7, 190.85, 186.22, 164.82, 162.93, 141.53, 187.58, 130.59, 
        201.28, 112.77, 232.79, 103.19, 261.55, 114.15, 261.55, 148.39, 260.17, 162.09, 332.77, 208.65, 357.42, 227.83, 386.19, 260.7, 351.94, 296.31, 328.66, 
        307.27, 314.97, 383.99, 280.72, 483.97, 256.07, 503.14, 306.75, 551.09, 327.29, 570.26, 367.01, 568.9, 356.06, 629.16, 314.97, 629.16, 305.37, 607.24, 
        254.71, 556.57, 238.27, 616.83, 234.15, 630.52, 193.06, 629.16]], 'area': 59028.103500000005, 'iscrowd': 0, 'image_id': 314385, 'bbox': [117.73, 103.19, 268.46, 527.33],
        'category_id': 1, 'id': 439182},
        {'segmentation': [[121.83, 266.73, 110.42, 302.76, 94.2, 336.39, 71.99, 395.84, 71.99, 405.45, 78.59, 409.65, 88.2, 405.45, 116.42, 352.61, 121.23, 323.78, 124.83, 297.36,
        131.44, 267.33, 126.03, 267.33, 124.23, 267.33]], 'area': 2781.5001000000007, 'iscrowd': 0, 'image_id': 314385, 'bbox': [71.99, 266.73, 59.45, 142.92], 'category_id': 43, 'id': 656817}]        
        '''
        
    def __len__(self):
        '''
        return len(self._labels)
        '''
        ####################################
        
        return self._num_samples
        
        ####################################
    ######################################################
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)  
        return bbox
        
    def _coco_box_center(self, box):
        center = np.array([box[0] + (box[2]/2), box[1] + (box[3]/2)], dtype=np.float32)  
        return center
    ######################################################    
    
    def _generate_keypoint_maps(self, sample):
        '''
        n_keypoints = 18
        n_rows, n_cols, _ = sample['image'].shape
        keypoint_maps = np.zeros(shape=(n_keypoints + 1,
                                        n_rows // self._stride, n_cols // self._stride), dtype=np.float32)  # +1 for bg

        label = sample['label']
        for keypoint_idx in range(n_keypoints):
            keypoint = label['keypoints'][keypoint_idx]
            if keypoint[2] <= 1:
                self._add_gaussian(keypoint_maps[keypoint_idx], keypoint[0], keypoint[1], self._stride, self._sigma)
            for another_annotation in label['processed_other_annotations']:
                keypoint = another_annotation['keypoints'][keypoint_idx]
                if keypoint[2] <= 1:
                    self._add_gaussian(keypoint_maps[keypoint_idx], keypoint[0], keypoint[1], self._stride, self._sigma)
        keypoint_maps[-1] = 1 - keypoint_maps.max(axis=0)
        return keypoint_maps
        '''
    ######################################################
        n_keypoints = 3 # * 80
        n_rows, n_cols, _ = sample['image'].shape
        keypoint_maps = np.zeros(shape=(n_keypoints + 1, n_rows // self._stride, n_cols // self._stride), dtype=np.float32)  # +1 for bg 
        
        label = sample['label']
        for bbox_ann in label:
            #bbox_class = bbox_ann['class_label']
            x1,y1,x2,y2 = bbox_ann['bbox']
            x3,y3 = bbox_ann['objpos']
            '''
            self._add_gaussian(keypoint_maps[bbox_class],x1,y1,self._stride, self._sigma)  # [0:background, 1-80:top_left, 81-160:middle, 161-240:bottom_right]
            self._add_gaussian(keypoint_maps[bbox_class+80],x3,y3,self._stride, self._sigma)
            self._add_gaussian(keypoint_maps[bbox_class+160],x2,y2,self._stride, self._sigma)
            '''
            self._add_gaussian(keypoint_maps[1],x1,y1,self._stride, self._sigma)  # [0:background, 1:top_left, 2:middle, 3:bottom_right]
            self._add_gaussian(keypoint_maps[2],x3,y3,self._stride, self._sigma)
            self._add_gaussian(keypoint_maps[3],x2,y2,self._stride, self._sigma)            
        keypoint_maps[0] = 1 - keypoint_maps.max(axis=0)
        return keypoint_maps
    ######################################################
    def _add_gaussian(self, keypoint_map, x, y, stride, sigma):
        n_sigma = 4
        tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
        tl[0] = max(tl[0], 0)
        tl[1] = max(tl[1], 0)

        br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
        map_h, map_w = keypoint_map.shape
        br[0] = min(br[0], map_w * stride)
        br[1] = min(br[1], map_h * stride)

        shift = stride / 2 - 0.5
        for map_y in range(tl[1] // stride, br[1] // stride):
            for map_x in range(tl[0] // stride, br[0] // stride):
                d2 = (map_x * stride + shift - x) * (map_x * stride + shift - x) + \
                    (map_y * stride + shift - y) * (map_y * stride + shift - y)
                exponent = d2 / 2 / sigma / sigma
                if exponent > 4.6052:  # threshold, ln(100), ~0.01
                    continue
                keypoint_map[map_y, map_x] += math.exp(-exponent)
                if keypoint_map[map_y, map_x] > 1:
                    keypoint_map[map_y, map_x] = 1

    def _generate_paf_maps(self, sample):
        '''
        n_pafs = len(BODY_PARTS_KPT_IDS)
        n_rows, n_cols, _ = sample['image'].shape
        paf_maps = np.zeros(shape=(n_pafs * 2, n_rows // self._stride, n_cols // self._stride), dtype=np.float32)

        label = sample['label']
        for paf_idx in range(n_pafs):
            keypoint_a = label['keypoints'][BODY_PARTS_KPT_IDS[paf_idx][0]]
            keypoint_b = label['keypoints'][BODY_PARTS_KPT_IDS[paf_idx][1]]
            if keypoint_a[2] <= 1 and keypoint_b[2] <= 1:
                self._set_paf(paf_maps[paf_idx * 2:paf_idx * 2 + 2],
                              keypoint_a[0], keypoint_a[1], keypoint_b[0], keypoint_b[1],
                              self._stride, self._paf_thickness)
            for another_annotation in label['processed_other_annotations']:
                keypoint_a = another_annotation['keypoints'][BODY_PARTS_KPT_IDS[paf_idx][0]]
                keypoint_b = another_annotation['keypoints'][BODY_PARTS_KPT_IDS[paf_idx][1]]
                if keypoint_a[2] <= 1 and keypoint_b[2] <= 1:
                    self._set_paf(paf_maps[paf_idx * 2:paf_idx * 2 + 2],
                                  keypoint_a[0], keypoint_a[1], keypoint_b[0], keypoint_b[1],
                                  self._stride, self._paf_thickness)
        return paf_maps
        '''
        ####################################################################
        n_pafs = 2 #*80    # (center to top left + center to bottom right) * 80 classes
        n_rows, n_cols, _ = sample['image'].shape
        paf_maps = np.zeros(shape=(n_pafs * 2, n_rows // self._stride, n_cols // self._stride), dtype=np.float32)  # [0-159: center -> topleft; 160-319: center -> bottomright] consecutive (x,y)
                                                                                                                   
        label = sample['label']
        for bbox_ann in label:
            #bbox_class = bbox_ann['class_label']
            x1,y1,x2,y2 = bbox_ann['bbox']
            x3,y3 = bbox_ann['objpos']
            '''
            self._set_paf(paf_maps[(bbox_class-1) * 2:(bbox_class-1) * 2 + 2],x3,y3,x1,y1,self._stride, self._paf_thickness) # first person class starts from 0
            self._set_paf(paf_maps[(bbox_class-1) * 2 + 160:(bbox_class-1) * 2 + 162],x3,y3,x2,y2,self._stride, self._paf_thickness) 
            '''
            self._set_paf(paf_maps[:2],x3,y3,x1,y1,self._stride, self._paf_thickness) # first person class starts from 0
            self._set_paf(paf_maps[2:4],x3,y3,x2,y2,self._stride, self._paf_thickness)             
        return paf_maps    
        #####################################################################
        
    def _set_paf(self, paf_map, x_a, y_a, x_b, y_b, stride, thickness):
        x_a /= stride
        y_a /= stride
        x_b /= stride
        y_b /= stride
        x_ba = x_b - x_a
        y_ba = y_b - y_a
        _, h_map, w_map = paf_map.shape
        x_min = int(max(min(x_a, x_b) - thickness, 0))
        x_max = int(min(max(x_a, x_b) + thickness, w_map))
        y_min = int(max(min(y_a, y_b) - thickness, 0))
        y_max = int(min(max(y_a, y_b) + thickness, h_map))
        norm_ba = (x_ba * x_ba + y_ba * y_ba) ** 0.5
        if norm_ba < 1e-7:  # Same points, no paf
            return
        x_ba /= norm_ba
        y_ba /= norm_ba

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                x_ca = x - x_a
                y_ca = y - y_a
                d = math.fabs(x_ca * y_ba - y_ca * x_ba)
                if d <= thickness:
                    paf_map[0, y, x] = x_ba
                    paf_map[1, y, x] = y_ba

class CocoValDataset(Dataset):
    def __init__(self, images_folder, stride, sigma, paf_thickness, transform=None):
        super().__init__()
        self._images_folder = images_folder
        self._stride = stride
        self._sigma = sigma
        self._paf_thickness = paf_thickness
        
        self._data_dir = '/home/hl3424@columbia.edu/PAF/openpose_pytorch/COCO/annotations/'
        self._annot_path =  os.path.join(self._data_dir,'instances_val2017.json')    
        self._max_objs = 128
        
        self._class_name = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self._valid_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
        24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
        37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
        58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
        82, 84, 85, 86, 87, 88, 89, 90]
        self._cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        
        print('==> initializing coco 2017 training data.')
        self._coco = coco.COCO(self._annot_path)
        self._images = self._coco.getImgIds()
        self._num_samples = len(self._images)

        print('Loaded training {} samples'.format(self._num_samples))

    def __getitem__(self, idx):
        img_id = self._images[idx]
        file_name = self._coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self._images_folder, file_name)
        ann_ids = self._coco.getAnnIds(imgIds=[img_id])
        anns = self._coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self._max_objs)   
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        height, width = img.shape[0], img.shape[1]
        
        transform = A.Compose([
        A.ShiftScaleRotate(),
        A.RandomSizedBBoxSafeCrop(height=368, width=368),
        A.HorizontalFlip(p=0.5)
        ], bbox_params=A.BboxParams(format='coco', min_area=16, min_visibility=0.05,label_fields=['class_labels']))  
        
        bboxes = []
        class_labels = []
        
        for ann in anns:
            if (ann['bbox'][2] >= 1) and (ann['bbox'][3] >= 1):  # handle xmax = xmin || ymax = ymin case
                bboxes.append(ann['bbox'])
                class_labels.append(int(self._cat_ids[ann['category_id']])+1)  # background is 0, person 1, ....    
        
        transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_class_labels = transformed['class_labels']
        
        label = []
        for idx,coco_box in enumerate(transformed_bboxes):
            bbox_center = self._coco_box_center(coco_box)
            bbox = self._coco_box_to_bbox(coco_box)
            bbox_ann = {}
            bbox_ann['objpos'] = bbox_center
            bbox_ann['bbox'] = bbox
            label.append(bbox_ann)
            
        image = transformed_image
        sample = {
            'label': label,
            'image': image,
        }      

        keypoint_maps = self._generate_keypoint_maps(sample)
        sample['keypoint_maps'] = keypoint_maps        
        paf_maps = self._generate_paf_maps(sample)
        sample['paf_maps'] = paf_maps
        image = sample['image'].astype(np.float32)          # image size (368,368) 
        image = (image - 128) / 256                         # pixel value range from -0.5 to 0.5
        sample['image'] = image.transpose((2, 0, 1))
        del sample['label']                                 # sample dict_keys(['image','keypoint_maps','paf_maps'])
        return sample        
    
        
    def __len__(self):
        return self._num_samples

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)  
        return bbox
        
    def _coco_box_center(self, box):
        center = np.array([box[0] + (box[2]/2), box[1] + (box[3]/2)], dtype=np.float32)  
        return center
    
    def _generate_keypoint_maps(self, sample):
        n_keypoints = 3 # * 80
        n_rows, n_cols, _ = sample['image'].shape
        keypoint_maps = np.zeros(shape=(n_keypoints + 1, n_rows // self._stride, n_cols // self._stride), dtype=np.float32)  # +1 for bg 
        
        label = sample['label']
        for bbox_ann in label:
            #bbox_class = bbox_ann['class_label']
            x1,y1,x2,y2 = bbox_ann['bbox']
            x3,y3 = bbox_ann['objpos']
            self._add_gaussian(keypoint_maps[1],x1,y1,self._stride, self._sigma)  # [0:background, 1:top_left, 2:middle, 3:bottom_right]
            self._add_gaussian(keypoint_maps[2],x3,y3,self._stride, self._sigma)
            self._add_gaussian(keypoint_maps[3],x2,y2,self._stride, self._sigma)            
        keypoint_maps[0] = 1 - keypoint_maps.max(axis=0)
        return keypoint_maps

    def _add_gaussian(self, keypoint_map, x, y, stride, sigma):
        n_sigma = 4
        tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
        tl[0] = max(tl[0], 0)
        tl[1] = max(tl[1], 0)

        br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
        map_h, map_w = keypoint_map.shape
        br[0] = min(br[0], map_w * stride)
        br[1] = min(br[1], map_h * stride)

        shift = stride / 2 - 0.5
        for map_y in range(tl[1] // stride, br[1] // stride):
            for map_x in range(tl[0] // stride, br[0] // stride):
                d2 = (map_x * stride + shift - x) * (map_x * stride + shift - x) + \
                    (map_y * stride + shift - y) * (map_y * stride + shift - y)
                exponent = d2 / 2 / sigma / sigma
                if exponent > 4.6052:  # threshold, ln(100), ~0.01
                    continue
                keypoint_map[map_y, map_x] += math.exp(-exponent)
                if keypoint_map[map_y, map_x] > 1:
                    keypoint_map[map_y, map_x] = 1

    def _generate_paf_maps(self, sample):
        n_pafs = 2 #*80    # (center to top left + center to bottom right) * 80 classes
        n_rows, n_cols, _ = sample['image'].shape
        paf_maps = np.zeros(shape=(n_pafs * 2, n_rows // self._stride, n_cols // self._stride), dtype=np.float32)  # [0-159: center -> topleft; 160-319: center -> bottomright] consecutive (x,y)
                                                                                                                   
        label = sample['label']
        for bbox_ann in label:
            #bbox_class = bbox_ann['class_label']
            x1,y1,x2,y2 = bbox_ann['bbox']
            x3,y3 = bbox_ann['objpos']
            self._set_paf(paf_maps[:2],x3,y3,x1,y1,self._stride, self._paf_thickness) # first person class starts from 0
            self._set_paf(paf_maps[2:4],x3,y3,x2,y2,self._stride, self._paf_thickness)             
        return paf_maps    
        
    def _set_paf(self, paf_map, x_a, y_a, x_b, y_b, stride, thickness):
        x_a /= stride
        y_a /= stride
        x_b /= stride
        y_b /= stride
        x_ba = x_b - x_a
        y_ba = y_b - y_a
        _, h_map, w_map = paf_map.shape
        x_min = int(max(min(x_a, x_b) - thickness, 0))
        x_max = int(min(max(x_a, x_b) + thickness, w_map))
        y_min = int(max(min(y_a, y_b) - thickness, 0))
        y_max = int(min(max(y_a, y_b) + thickness, h_map))
        norm_ba = (x_ba * x_ba + y_ba * y_ba) ** 0.5
        if norm_ba < 1e-7:  # Same points, no paf
            return
        x_ba /= norm_ba
        y_ba /= norm_ba

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                x_ca = x - x_a
                y_ca = y - y_a
                d = math.fabs(x_ca * y_ba - y_ca * x_ba)
                if d <= thickness:
                    paf_map[0, y, x] = x_ba
                    paf_map[1, y, x] = y_ba
                    
class CocoTestDataset(Dataset):
    def __init__(self, images_folder):
        super().__init__()
        '''
        with open(labels, 'r') as f:
            self._labels = json.load(f)
        '''
        
        self._images_folder = images_folder
        self._data_dir = '/home/hl3424@columbia.edu/PAF/openpose_pytorch/COCO/annotations/'
        self._annot_path =  os.path.join(self._data_dir,'instances_val2017.json')
        self._coco = coco.COCO(self._annot_path)
        self._images = self._coco.getImgIds()
        self._num_samples = len(self._images)
        print('Loaded test {} samples'.format(self._num_samples))    
        
    def __getitem__(self, idx):
        '''
        file_name = self._labels['images'][idx]['file_name']
        img = cv2.imread(os.path.join(self._images_folder, file_name), cv2.IMREAD_COLOR)
        return {
            'img': img,
            'file_name': file_name
        }
        '''
        img_id = self._images[idx]
        file_name = self._coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self._images_folder, file_name)
        
        ann_ids = self._coco.getAnnIds(imgIds=[img_id])
        anns = self._coco.loadAnns(ids=ann_ids)
        self._max_objs = 128
        num_objs = min(len(anns), self._max_objs)   
        bboxes = []
        for ann in anns:
            if (ann['bbox'][2] >= 1) and (ann['bbox'][3] >= 1):  # handle xmax = xmin || ymax = ymin case
                bboxes.append(ann['bbox']) 
        label = []
        for idx,coco_box in enumerate(bboxes):
            bbox_center = self._coco_box_center(coco_box)
            bbox = self._coco_box_to_bbox(coco_box)
            bbox_ann = {}
            bbox_ann['objpos'] = bbox_center
            bbox_ann['bbox'] = bbox
            label.append(bbox_ann)         
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        height, width = img.shape[0], img.shape[1]
        
        sample = {
            'image': img,
            'file_name': file_name,
            'label': label
        }  
        
        return sample    
        
    def __len__(self):
        '''
        return len(self._labels)
        '''
        ####################################
        
        return self._num_samples
        
        ####################################                    
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)  
        return bbox
        
    def _coco_box_center(self, box):
        center = np.array([box[0] + (box[2]/2), box[1] + (box[3]/2)], dtype=np.float32)  
        return center