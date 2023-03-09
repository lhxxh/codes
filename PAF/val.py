import argparse
import cv2
import json
import math
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch
import time

from datasets.coco import CocoTestDataset
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state

import json

def run_coco_eval(gt_file_path, dt_file_path):
    annotation_type = 'keypoints'
    print('Running test for {} results.'.format(annotation_type))

    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(dt_file_path)

    result = COCOeval(coco_gt, coco_dt, annotation_type)
    result.evaluate()
    result.accumulate()
    result.summarize()


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


def convert_to_coco_format(pose_entries, all_keypoints):
    coco_keypoints = []
    scores = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        keypoints = [0] * 17 * 3
        to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        person_score = pose_entries[n][-2]
        position_id = -1
        for keypoint_id in pose_entries[n][:-2]:
            position_id += 1
            if position_id == 1:  # no 'neck' in COCO
                continue

            cx, cy, score, visibility = 0, 0, 0, 0  # keypoint not found
            if keypoint_id != -1:
                cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                cx = cx + 0.5
                cy = cy + 0.5
                visibility = 1
            keypoints[to_coco_map[position_id] * 3 + 0] = cx
            keypoints[to_coco_map[position_id] * 3 + 1] = cy
            keypoints[to_coco_map[position_id] * 3 + 2] = visibility
        coco_keypoints.append(keypoints)
        scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'
    return coco_keypoints, scores


def infer(net, img, scales, base_height, stride, pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    normed_img = normalize(img, img_mean, img_scale)
    height, width, _ = normed_img.shape
    scales_ratios = [scale * base_height / float(height) for scale in scales]
    #avg_heatmaps = np.zeros((height, width, 19), dtype=np.float32)
    #avg_pafs = np.zeros((height, width, 38), dtype=np.float32)
    #######################################################################
    avg_heatmaps = np.zeros((height, width, 4), dtype=np.float32)
    avg_pafs = np.zeros((height, width, 4), dtype=np.float32)    
    #######################################################################

    for ratio in scales_ratios:
        scaled_img = cv2.resize(normed_img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        min_dims = [base_height, max(scaled_img.shape[1], base_height)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)   

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
        stages_output = net(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))   # compatible with (height,width,dimension)
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)  #expand heatmaps by stride
        heatmaps = heatmaps[pad[0]:heatmaps.shape[0] - pad[2], pad[1]:heatmaps.shape[1] - pad[3]:, :] # extract heatmaps from padding
        heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_heatmaps = avg_heatmaps + heatmaps / len(scales_ratios)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        pafs = pafs[pad[0]:pafs.shape[0] - pad[2], pad[1]:pafs.shape[1] - pad[3], :]
        pafs = cv2.resize(pafs, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_pafs = avg_pafs + pafs / len(scales_ratios)

    return avg_heatmaps, avg_pafs


def evaluate(labels, output_name, images_folder, net, multiscale=False, visualize=False):
    net = net.cuda().eval()
    base_height = 368
    scales = [1]
    if multiscale:
        scales = [0.5, 1.0, 1.5, 2.0]
    stride = 8

    #dataset = CocoValDataset(labels, images_folder)
    ###########################################################
    dataset = CocoTestDataset(images_folder)
    ############################################################
    coco_result = []
    
    print('______total_length______',len(dataset))
    averaged_infer_time = []
    ret_list = []
    for curr_idx,sample in enumerate(dataset):
        if curr_idx % 100 == 0:
            print(curr_idx)
        file_name = sample['file_name']
        img = sample['image']
        label = sample['label']

        infer_start = time.time()
        avg_heatmaps, avg_pafs = infer(net, img, scales, base_height, stride)  # inference; output averaged over different ratios
        infer_end = time.time()
        infer_time = infer_end - infer_start
        averaged_infer_time.append(infer_time)
        
        '''
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(18):  # 19th for bg
            total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)

        coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)
        '''
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(4):  
            total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        box_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)   # postprocessing part
        
        '''
        _class_name = [
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
        '''
        ret_sample = {}
        ret_sample['file_name'] = file_name
        box_entries_list = []
        box_score_list = []
        for n in range(len(box_entries)):
            tlx = -1
            tly = -1
            cx  = -1
            cy  = -1
            brx = -1
            bry = -1       
            score = 0
            for idx,keypoint_id in enumerate(box_entries[n]):
                if (idx%3 == 0) and (keypoint_id != -1):
                    x, y, score1 = all_keypoints[int(keypoint_id), 0:3]
                    tlx = int(x)
                    tly = int(y)
                    score += score1
                elif (idx%3 == 1) and (keypoint_id != -1):
                    x, y, score2 = all_keypoints[int(keypoint_id), 0:3]
                    cx = int(x)
                    cy = int(y)
                    score += score2
                elif (idx%3 == 2) and (keypoint_id != -1):
                    x, y, score3 = all_keypoints[int(keypoint_id), 0:3]
                    brx = int(x)
                    bry = int(y)
                    score += score3
                #else:
                #    class_ = keypoint_id
                
                
            if (tlx == -1) and (tly  == -1):
                xdiff = brx - cx
                ydiff = bry - cy
                tlx = cx - xdiff
                if tlx < 0:
                    tlx = 0
                tly = cy - ydiff
                if tly < 0:
                    tly = 0                       
            elif (brx == -1) and (bry == -1):
                xdiff = cx - tlx
                ydiff = cy - tly
                brx = cx + xdiff
                if brx >= img.shape[0]:
                    brx = img.shape[0]-1
                bry = cy + ydiff
                if bry >= img.shape[1]:
                    bry = img.shape[1]-1
            
            tlx_normalized = (tlx/img.shape[1])
            tly_normalized = (tly/img.shape[0])
            brx_normalized = (brx/img.shape[1])
            bry_normalized = (bry/img.shape[0])
            
            box_entries_list.append([str(tlx_normalized),str(tly_normalized),str(brx_normalized),str(bry_normalized)])
            box_score_list.append(str(score/3))            
        ret_sample['normalized_bbox_coordinate'] = box_entries_list
        ret_sample['bbox_score'] = box_score_list
        ret_list.append(ret_sample)
        
        if visualize:
            gt_img = np.copy(img)
            for bbox_ann in label:
                gt_bbox = bbox_ann['bbox']
                cv2.rectangle(gt_img,(gt_bbox[0],gt_bbox[1]),(gt_bbox[2],gt_bbox[3]),(255,0,0),2)
            cv2.imwrite('data/results_alg2/gt/{}_gt.jpg'.format(curr_idx),gt_img)
            #### algorithm 1#####
            '''
            for n in range(len(box_entries)):
                tlx = -1
                tly = -1
                cx  = -1
                cy  = -1
                brx = -1
                bry = -1
                for idx,keypoint_id in enumerate(box_entries[n]):
                    if idx%4 == 0:
                        x, y, score = all_keypoints[int(keypoint_id), 0:3]
                        tlx = int(x)
                        tly = int(y)
                    elif idx%4 == 1:
                        x, y, score = all_keypoints[int(keypoint_id), 0:3]
                        cx = int(x)
                        cy = int(y)
                    elif idx%4 == 2:
                        x, y, score = all_keypoints[int(keypoint_id), 0:3]
                        brx = int(x)
                        bry = int(y)
                    else:
                        class_ = keypoint_id
                cv2.rectangle(img,(tlx,tly),(brx,bry),(0,0,255),2)  
                cv2.circle(img, (cx,cy), 5, (0, 0, 255), -1)    
                cv2.putText(img,_class_name[class_],(tlx,bry),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
                cv2.imwrite('data/results_alg1/{}.jpg'.format(file_name),img)
            '''
            for n in range(len(box_entries)):
                tlx = -1
                tly = -1
                cx  = -1
                cy  = -1
                brx = -1
                bry = -1       
                for idx,keypoint_id in enumerate(box_entries[n]):
                    if (idx%3 == 0) and (keypoint_id != -1):
                        x, y, score = all_keypoints[int(keypoint_id), 0:3]
                        tlx = int(x)
                        tly = int(y)
                    elif (idx%3 == 1) and (keypoint_id != -1):
                        x, y, score = all_keypoints[int(keypoint_id), 0:3]
                        cx = int(x)
                        cy = int(y)
                    elif (idx%3 == 2) and (keypoint_id != -1):
                        x, y, score = all_keypoints[int(keypoint_id), 0:3]
                        brx = int(x)
                        bry = int(y)
                    #else:
                    #    class_ = keypoint_id
                
                
                if (tlx == -1) and (tly  == -1):
                    xdiff = brx - cx
                    ydiff = bry - cy
                    tlx = cx - xdiff
                    if tlx < 0:
                        tlx = 0
                    tly = cy - ydiff
                    if tly < 0:
                        tly = 0                       
                elif (brx == -1) and (bry == -1):
                    xdiff = cx - tlx
                    ydiff = cy - tly
                    brx = cx + xdiff
                    if brx >= img.shape[0]:
                        brx = img.shape[0]-1
                    bry = cy + ydiff
                    if bry >= img.shape[1]:
                        bry = img.shape[1]-1
                    
                assert (tlx >= 0)
                assert (tly >= 0)
                assert (cx >= 0)
                assert (cy >= 0)
                assert (brx >= 0)
                assert (bry >= 0)                
                    
                cv2.rectangle(img,(tlx,tly),(brx,bry),(0,0,255),2)  
                cv2.circle(img, (cx,cy), 5, (0, 0, 255), -1)    
                #cv2.putText(img,_class_name[class_],(tlx,bry),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
            cv2.imwrite('data/results_alg2/0.8/{}.jpg'.format(curr_idx),img)     

    print('average infer time : {}'.format(sum(averaged_infer_time) / len(averaged_infer_time)))     
    with open("PAF_test_output.json", "w") as final:
        json.dump(ret_list, final)
    
'''        
        image_id = int(file_name[0:file_name.rfind('.')])
        for idx in range(len(coco_keypoints)):
            coco_result.append({
                'image_id': image_id,
                'category_id': 1,  # person
                'keypoints': coco_keypoints[idx],
                'score': scores[idx]
            })

        if visualize:
            for keypoints in coco_keypoints:
                for idx in range(len(keypoints) // 3):
                    cv2.circle(img, (int(keypoints[idx * 3]), int(keypoints[idx * 3 + 1])),
                               3, (255, 0, 255), -1)
            cv2.imshow('keypoints', img)
            key = cv2.waitKey()
            if key == 27:  # esc
                return
    
    with open(output_name, 'w') as f:
        json.dump(coco_result, f, indent=4)
    
    run_coco_eval(labels, output_name)
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, required=True, help='path to json with keypoints val labels')
    parser.add_argument('--output-name', type=str, default='detections.json',
                        help='name of output json file with detected keypoints')
    parser.add_argument('--images-folder', type=str, required=True, help='path to COCO val images folder')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--multiscale', action='store_true', help='average inference results over multiple scales')
    parser.add_argument('--visualize', action='store_true', help='show keypoints')
    args = parser.parse_args()

    #net = PoseEstimationWithMobileNet()
    ################################################################################
    net = PoseEstimationWithMobileNet(num_refinement_stages=1, num_heatmaps=4, num_pafs=4) 
    #################################################################################
    checkpoint = torch.load(args.checkpoint_path)
    load_state(net, checkpoint)

    evaluate(args.labels, args.output_name, args.images_folder, net, args.multiscale, args.visualize)
