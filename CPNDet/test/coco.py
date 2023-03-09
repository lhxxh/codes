# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import sys
import cv2
import pdb
import time
import json
import copy
import math
import torch
import matplotlib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import system_configs
from utils import crop_image, normalize_
from models.py_utils import bbox_overlaps
from PIL import Image, ImageDraw, ImageFont
from torch.multiprocessing import Process, Queue
from external.nms import soft_nms, soft_nms_merge

import pycocotools
import pycocotools.coco as coco

import copy

colours = np.random.rand(80,3)

def _rescale_dets(detections, ratios, borders, sizes):
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]  # (1,num_dets=2000,2)  (1,num_dets=2000,2)
    xs    /= ratios[:, 1][:, None, None] # ratios.shape (1,2)  xs:(1,num_dets=2000,2)
    ys    /= ratios[:, 0][:, None, None] # (1,num_dets=2000,2)
    xs    -= borders[:, 2][:, None, None]# border.shape (1,4)  border example [[ 15. 495.  63. 703.]]  
    ys    -= borders[:, 0][:, None, None]
    tx_inds = xs[:,:,0] <= -5
    bx_inds = xs[:,:,1] >= sizes[0,1]+5  #sizes.shape(1,2) sizes example [[480. 640.]]
    ty_inds = ys[:,:,0] <= -5
    by_inds = ys[:,:,1] >= sizes[0,0]+5   #all inds (1,num_dets=2000)
    
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)
    detections[:,tx_inds[0,:],4] = -1
    detections[:,bx_inds[0,:],4] = -1
    detections[:,ty_inds[0,:],4] = -1
    detections[:,by_inds[0,:],4] = -1

def kp_decode(nnet, images, K, no_flip, ae_threshold=0.5, kernel=3, image_idx = 0):
    detections = nnet.test([images], ae_threshold=ae_threshold, K=K, no_flip = no_flip, kernel=kernel, image_idx = image_idx)
    detections = detections.data.cpu().numpy()
    return detections

def _coco_box_to_bbox(box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)  
    return bbox
        
def _coco_box_center(box):
    center = np.array([box[0] + (box[2]/2), box[1] + (box[3]/2)], dtype=np.float32)  
    return center

def image_preprocess(db, cfg_file, db_inds, scales, result_dir, debug, no_flip, im_queue):
    num_images = db_inds.size
    
    for ind in range(0, num_images):
        db_ind = db_inds[ind]  

        image_id   = db.image_ids(db_ind)   # load image info(image id, image file location, image itself，height, width)
        image_file = db.image_file(db_ind)
        image      = cv2.imread(image_file)

        height, width = image.shape[0:2]  

        for scale in scales:                # if not multi-scale, scales = [1]; if multiscale, scales = [0.6, 1, 1.2, 1.5, 1.8]
            new_height = int(height * scale)  
            new_width  = int(width * scale)
            new_center = np.array([new_height // 2, new_width // 2])
            
            if 'DLA' in cfg_file:           # make sure the lower bound of cropped input images
                inp_height = (new_height | 31)+1
                inp_width  = (new_width | 31)+1
            else:
                inp_height = new_height | 127
                inp_width  = new_width | 127

            images  = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
            ratios  = np.zeros((1, 2), dtype=np.float32)
            borders = np.zeros((1, 4), dtype=np.float32)
            sizes   = np.zeros((1, 2), dtype=np.float32)
            
            if 'DLA' in cfg_file:         # keep ratio around (output size : input size) 1:4
                out_height, out_width = inp_height // 4, inp_width // 4
            else:
                out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
            
            height_ratio = out_height / inp_height
            width_ratio  = out_width  / inp_width

            resized_image = cv2.resize(image, (new_width, new_height))  # resize input image to scaled size
            resized_image, border, offset = crop_image(resized_image, new_center, [inp_height, inp_width]) # crop image (resized_image) around center(new_center) with shape(inp_height, inp_width) 

            resized_image = resized_image / 255.  # set pixel value within 0-1 float
            normalize_(resized_image, db.mean, db.std)  # normalize the resized input by whole dataset avg and std

            images[0]  = resized_image.transpose((2, 0, 1))  # tranpose to (batch,dim,width,height) 
            borders[0] = border
            sizes[0]   = [int(height * scale), int(width * scale)]
            ratios[0]  = [height_ratio, width_ratio]   

            if not no_flip:
                images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)  # add flipped image
            images = torch.from_numpy(images)
            
            im_queue.put([images, ratios, borders, sizes, out_width, image_id])
            
    time.sleep(num_images*10)
    
def post_process(db, debug, num_images, weight_exp, merge_bbox, categories, 
            nms_threshold, max_per_image, nms_algorithm, det_queue, top_bboxes_queue,db_inds): 
    top_bboxes = {}
    #######################################################
    ret_list = {}
    _data_dir = '/home/hl3424@columbia.edu/CPNDet/code/data/coco/annotations/'
    _annot_path =  os.path.join(_data_dir,'instances_minival2014.json')      # using validation set!!!!!
    _coco = coco.COCO(_annot_path)
    #######################################################
    for ind in range(0, num_images):
        det_bboxes = det_queue.get(block=True)
        detections = det_bboxes[0]
        classes = det_bboxes[1]
        image_id = det_bboxes[2]
        
        top_bboxes[image_id] = {}
        ret_list[image_id] = []
        for j in range(categories):
            keep_inds = (classes == j)
            top_bboxes[image_id][j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
            if merge_bbox:
                soft_nms_merge(top_bboxes[image_id][j + 1], Nt=nms_threshold, method=nms_algorithm, weight_exp=weight_exp)
            else:
                soft_nms(top_bboxes[image_id][j + 1], Nt=nms_threshold, method=nms_algorithm)
            top_bboxes[image_id][j + 1] = top_bboxes[image_id][j + 1][:, 0:5]

        scores = np.hstack([
            top_bboxes[image_id][j][:, -1] 
            for j in range(1, categories + 1)
        ])
        if len(scores) > max_per_image:
            kth    = len(scores) - max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, categories + 1):
                keep_inds = (top_bboxes[image_id][j][:, -1] >= thresh)
                top_bboxes[image_id][j] = top_bboxes[image_id][j][keep_inds]
        
        #############################################################################
        #scale the output
        image_file = db.image_file(ind)
        image      = cv2.imread(image_file)        

        for j in range(1, categories + 1):
            keep_inds = (top_bboxes[image_id][j][:, -1] >= 0.8)         #change threshold  
            cat_name  = db.class_name(j)

            for tt,bbox in enumerate(top_bboxes[image_id][j][keep_inds]):
                bbox_score = bbox[4]
                bbox  = bbox[0:4].astype(np.int32)
                xmin_s     = bbox[0]/image.shape[1]
                ymin_s     = bbox[1]/image.shape[0]
                xmax_s     = bbox[2]/image.shape[1]
                ymax_s     = bbox[3]/image.shape[0]
                bbox_score_c =  bbox_score /2
                ret_list[image_id].append([str(xmin_s),str(ymin_s),str(xmax_s),str(ymax_s),str(bbox_score_c)])
        ##############################################################################
                
        if debug:
            image_file = db.image_file(ind)
            image      = cv2.imread(image_file)
            
            #########################################################
            img_id = int((os.path.splitext(os.path.basename(db.image_file(ind)))[0]).rpartition("_")[-1])
            ann_ids = _coco.getAnnIds(imgIds=[img_id])
            anns = _coco.loadAnns(ids=ann_ids)  
            bboxes = []
            
            for ann in anns:
                if (ann['bbox'][2] >= 1) and (ann['bbox'][3] >= 1):  # handle xmax = xmin || ymax = ymin case
                    bboxes.append(ann['bbox'])
            
            label = []
            for idx,coco_box in enumerate(bboxes):
                bbox_center = _coco_box_center(coco_box)
                bbox = _coco_box_to_bbox(coco_box)
                bbox_ann = {}
                bbox_ann['objpos'] = bbox_center
                bbox_ann['bbox'] = bbox  
                label.append(bbox_ann)
            
            gt_img = np.copy(image)
            for bbox_ann in label:
                gt_bbox = bbox_ann['bbox']
                cv2.rectangle(gt_img,(int(gt_bbox[0]),int(gt_bbox[1])),(int(gt_bbox[2]),int(gt_bbox[3])),(255,0,0),2)
            cv2.imwrite('validations/gt/{}_gt.jpg'.format(db_inds[ind]),gt_img)            
            #########################################################
            
            im         = image[:, :, (2, 1, 0)]
            fig, ax    = plt.subplots(figsize=(12, 12)) 
            fig        = ax.imshow(im, aspect='equal')
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            bboxes = {}
            for j in range(1, categories + 1):
                keep_inds = (top_bboxes[image_id][j][:, -1] >= 0.5)         #change threshold  
                cat_name  = db.class_name(j)
                for bbox in top_bboxes[image_id][j][keep_inds]:
                    bbox_score = bbox[4]
                    bbox  = bbox[0:4].astype(np.int32)
                    xmin     = bbox[0]
                    ymin     = bbox[1]
                    xmax     = bbox[2]
                    ymax     = bbox[3]
                    if (xmax - xmin) * (ymax - ymin) > 100:
                        ax.add_patch(plt.Rectangle((xmin, ymin),xmax - xmin, ymax - ymin, fill=False, edgecolor= 'red', 
                                linewidth=2.0))
                        #ax.text(xmin+1, ymin-3, '{:s} {:2f}'.format(cat_name,bbox_score), bbox=dict(facecolor= colours[j-1], ec='black', lw=2,alpha=0.5),    # add bbox score here 
                        #   fontsize=15, color='white', weight='bold')

            debug_file1 = os.path.join("validations/0.5/{}.pdf".format(db_inds[ind]))
            debug_file2 = os.path.join("validations/0.5/{}.jpg".format(db_inds[ind]))
            plt.savefig(debug_file1)
            plt.savefig(debug_file2)
            plt.close()
            cv2.imwrite(debug_file2, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    
    top_bboxes_queue.put([top_bboxes,ret_list])
    
def kp_detection(db, cfg_file, nnet, result_dir, debug=False, no_flip = False, decode_func=kp_decode):
    image_idx = 0
    debug_dir = os.path.join(result_dir, "debug")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    if db.split != "trainval": # if debug, first 100 pictures; if not debug + trainval, whole dataset; if not debug + test, first 5000 pictures
        db_inds = db.db_inds[:100] if debug else db.db_inds  
    else:
        db_inds = db.db_inds[:100] if debug else db.db_inds[:5000]
    num_images  = db_inds.size

    K            = db.configs["top_k"]
    ae_threshold = db.configs["ae_threshold"]
    nms_kernel   = db.configs["nms_kernel"]
    
    scales        = db.configs["test_scales"]
    weight_exp    = db.configs["weight_exp"]
    merge_bbox    = db.configs["merge_bbox"]
    categories    = db.configs["categories"]
    nms_threshold = db.configs["nms_threshold"]
    max_per_image = db.configs["max_per_image"]
    nms_algorithm = {
        "nms": 0,
        "linear_soft_nms": 1, 
        "exp_soft_nms": 2
    }[db.configs["nms_algorithm"]]
    
    im_queue = Queue()
    det_queue = Queue()
    top_bboxes_queue = Queue()
    
    im_process_task = Process(target=image_preprocess, args=(db, cfg_file, db_inds, scales, result_dir, debug, no_flip, im_queue))
    post_process_task = Process(target=post_process, args=(db, debug, num_images, weight_exp, merge_bbox, categories, nms_threshold, 
                                         max_per_image, nms_algorithm, det_queue, top_bboxes_queue,db_inds))
    
    im_process_task.start()
    post_process_task.start()
    
    start = time.time()
    for ind in tqdm(range(0, num_images), ncols=80, desc="locating kps"):
        detections = []
        detections_ori   = []
        detections_flip  = []
        for scale in scales: #if not multi-scale, scales = [1]; if multiscale, scales = [0.6, 1, 1.2, 1.5, 1.8]
            pre_data  = im_queue.get(block=True)  # decode the pre-processed images
            images    = pre_data[0]
            ratios    = pre_data[1]
            borders   = pre_data[2]
            sizes     = pre_data[3]
            out_width = pre_data[4]
            image_id  = pre_data[5]
            dets = decode_func(nnet, images, K, no_flip, ae_threshold=ae_threshold, kernel=nms_kernel, image_idx = image_idx) # feed forward through the network
            image_idx += 1
            if no_flip:
                dets   = dets.reshape(1, -1, 8)
                _rescale_dets(dets, ratios, borders, sizes)
                dets[:, :, 0:4] /= scale
                detections.append(dets)
            else:
                dets   = dets.reshape(2, -1, 8)   # [bboxes, scores, tl_scores, br_scores, clses]  bboxes: (batch,K*K,4)  other: (batch,K*K,1)  (2,num_dets=1000,8)
                dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]   # flip the flipped-image back
                dets   = dets.reshape(1, -1, 8)   # (1,num_dets=2000,8)
                _rescale_dets(dets, ratios, borders, sizes)
                dets[:, :, 0:4] /= scale
                detections_ori.append(dets[:,:int(dets.shape[1]/2),:])
                detections_flip.append(dets[:,int(dets.shape[1]/2):,:])
         
        if no_flip:
            detections = np.concatenate(detections, axis=1)
            classes    = detections[..., -1]
            classes    = classes[0]#[int(detections.shape[1]/2):]
            detections = detections[0]#[int(detections.shape[1]/2):]

            # reject detections with negative scores
            keep_inds  = (detections[:, 4] > 0)                   
            detections = detections[keep_inds]
            classes    = classes[keep_inds]
        else:    
            detections_ori_ = np.concatenate(detections_ori, axis=1)
            detections_flip_= np.concatenate(detections_flip, axis=1)

            detections  = np.concatenate((detections_ori_, detections_flip_), axis=1)
        
            detections1 = detections[0][:int(detections.shape[1]/2),:]
            detections2 = detections[0][int(detections.shape[1]/2):,:]

            keep_inds1  = (detections1[:, 4] > 0)               
            keep_inds2  = (detections2[:, 4] > 0)              

            detections_G1 = torch.from_numpy(detections1[keep_inds1]).cuda()
            detections_G2 = torch.from_numpy(detections2[keep_inds2]).cuda()

            detections_G1[:,4] = 0
            detections_G2[:,4] = 0

            detections1_matrix = detections_G1.permute(1,0).unsqueeze(-1).expand(8, detections_G1.size(0), 
                                                            detections_G2.size(0)).contiguous()
            detections2_matrix = detections_G2.permute(1,0).unsqueeze(1).expand(8, detections_G1.size(0), 
                                                            detections_G2.size(0)).contiguous()

            cls_inds = (detections1_matrix[-1,...] == detections2_matrix[-1, ...])

            select_detections1 = detections1_matrix[:4, cls_inds].permute(1,0).contiguous()
            select_detections2 = detections2_matrix[:4, cls_inds].permute(1,0).contiguous()

            overlaps = bbox_overlaps(select_detections1, select_detections2, is_aligned = True)
            if overlaps.size(0) > 0:
                detections1_conf = overlaps
                detections2_conf = overlaps

                detections1_matrix[4, cls_inds] = detections1_conf
                detections2_matrix[4, cls_inds] = detections2_conf

                detections1_conf_max = detections1_matrix[4,:,:].max(1)[0]
                detections2_conf_max = detections2_matrix[4,:,:].max(0)[0]

                conf_max = torch.cat([detections1_conf_max,detections2_conf_max], dim = 0).data.cpu().numpy()
                conf_max[conf_max<0.3] = 0
                ##################################################################################
            classes    = detections[..., -1]
            classes    = classes[0]#[int(detections.shape[1]/2):]
            detections = detections[0]#[int(detections.shape[1]/2):]

            # reject detections with negative scores
            keep_inds  = (detections[:, 4] > 0)                       
            detections = detections[keep_inds]
            classes    = classes[keep_inds]

            if overlaps.size(0) > 0:
                detections[:,4] += detections[:,4] * conf_max
                keep_inds  = (detections[:, 4] > 0)                     
                detections = detections[keep_inds]
                classes    = classes[keep_inds]
            
        det_queue.put([detections, classes, image_id])
    
    top_bboxes,ret_list = top_bboxes_queue.get(block=True)   

    ###########################################################################################
    with open("CPNDet_test_output.json", "w") as final:
        json.dump(ret_list, final)    
    ###########################################################################################                         
    
    elapsed = time.time() - start
    print('Average FPS: {}\n'.format(round(num_images/elapsed, 2)))
    
    im_process_task.terminate()
    post_process_task.terminate()
    
    result_json = os.path.join(result_dir, "results.json")
    detections  = db.convert_to_coco(top_bboxes)
    with open(result_json, "w") as f:
        json.dump(detections, f)
        
    cls_ids   = list(range(1, categories + 1))
    image_ids = [db.image_ids(ind) for ind in db_inds]
    db.evaluate(result_json, cls_ids, image_ids)    

    return 0

def testing(db, cfg_file, nnet, result_dir, debug=False, no_flip = False):
    return globals()[system_configs.sampling_function](db, cfg_file, nnet, result_dir, debug=debug, no_flip=no_flip)
