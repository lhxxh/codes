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

import pdb
import torch
import numpy as np
from . import builder
import torch.nn as nn
from mmcv import Config
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import time
import sys

from .utils import convolution, residual
from .utils import make_layer, make_layer_revr
from .kp_utils import make_pool_layer, make_unpool_layer
from .bbox import build_assigner, build_sampler, bbox2roi
from .kp_utils import make_merge_layer, make_inter_layer, make_cnv_layer
from .kp_utils import _sigmoid, _ae_loss, _regr_loss, _neg_loss, bbox_overlaps
from .kp_utils import make_tl_layer, make_br_layer, make_region_layer, make_kp_layer, _regr_l1_loss
from .kp_utils import _tranpose_and_gather_feat, _decode, _generate_bboxes, _htbox2roi, _htbox2roi_test, _filter_bboxes, center_filtering_test, center_filtering_train

class kp_module(nn.Module):
    def __init__(
        self, n, dims, modules, layer=residual,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, **kwargs
    ):
        super(kp_module, self).__init__()

        self.n   = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1  = make_up_layer(
            3, curr_dim, curr_dim, curr_mod, 
            layer=layer, **kwargs
        )  
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.low2 = kp_module(
            n - 1, dims[1:], modules[1:], layer=layer, 
            make_up_layer=make_up_layer, 
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
        make_low_layer(
            3, next_dim, next_dim, next_mod,
            layer=layer, **kwargs
        )
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.up2  = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return self.merge(up1, up2)

class hg52(nn.Module):
    def __init__(
        self, db, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256, 
        make_tl_layer=make_tl_layer, make_br_layer=make_br_layer,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_grouping_layer = make_region_layer, make_regr_layer=make_kp_layer,
        make_region_layer = make_region_layer, make_up_layer=make_layer, make_low_layer=make_layer, 
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer, 
        kp_layer=residual
    ):
        super(hg52, self).__init__()

        self.nstack           = nstack
        self._decode          = _decode
        self._generate_bboxes = _generate_bboxes
        self._db              = db
        self.K                = self._db.configs["top_k"]
        self.input_size       = db.configs["input_size"]
        self.output_size      = db.configs["output_sizes"][0]
        self.kernel           = self._db.configs["nms_kernel"]
        self.gr_threshold     = self._db.configs["gr_threshold"]
        self.categories       = self._db.configs["categories"]
        
        self.grouping_roi_extractor = builder.build_roi_extractor(Config(self._db._model['grouping_roi_extractor']).item)
        self.region_roi_extractor   = builder.build_roi_extractor(Config(self._db._model['region_roi_extractor']).item)
        
        self.roi_out_size   = Config(self._db._model['grouping_roi_extractor']).item.roi_layer.out_size
        self.iou_threshold  = self._db.configs["iou_threshold"]
        self.train_cfg      = Config(self._db._model['train_cfg'])
        self.bbox_head      = builder.build_bbox_head(Config(self._db._model['bbox_head']).item)
        
        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre   # make sure input : output = 4:1 currently

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])
        
        self.tl_cnvs = nn.ModuleList([
            make_tl_layer(cnv_dim) for _ in range(nstack)
        ])
        self.br_cnvs = nn.ModuleList([
            make_br_layer(cnv_dim) for _ in range(nstack)
        ])

        ## keypoint heatmaps
        self.tl_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.br_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        
        #################################################################
        self.ct_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])        
        #################################################################

        self.regions = nn.ModuleList([
            make_region_layer(cnv_dim, curr_dim) for _ in range(nstack)
         ])
        
        self.region_reduces = nn.ModuleList([
                          nn.Sequential(
                              nn.Conv2d(curr_dim, curr_dim, (self.roi_out_size, self.roi_out_size), bias=False),
                              nn.BatchNorm2d(curr_dim),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(curr_dim, out_dim, (1, 1))
                          ) for _ in range(nstack)
                       ])
        
        self.groupings = nn.ModuleList([
            make_grouping_layer(cnv_dim, 32) for _ in range(nstack)
         ])
        
        self.grouping_reduces = nn.ModuleList([
                          nn.Sequential(
                              nn.Conv2d(32, 32, (self.roi_out_size, self.roi_out_size), bias=False),
                              nn.BatchNorm2d(32),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(32, 1, (1, 1))
                          ) for _ in range(nstack)
                       ])

        for tl_heat, br_heat, region_reduce, grouping_reduce in zip \
           (self.tl_heats, self.br_heats, self.region_reduces, self.grouping_reduces):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)
            region_reduce[-1].bias.data.fill_(-2.19)
            grouping_reduce[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.tl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.br_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        
        #################################################################
        self.ct_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        #################################################################
        
        self.relu = nn.ReLU(inplace=True)
        
        self.init_weights()

    def init_weights(self):
        self.grouping_roi_extractor.init_weights()
        self.region_roi_extractor.init_weights()
        
    def _train(self, *xs):
        image         = xs[0]     #(batch,3,256,256)
        tl_inds       = xs[1]     #(batch,Max_target = 128)
        br_inds       = xs[2]     #(batch,Max_target = 128)
        gt_detections = xs[3]     #(batch,Max_target = 128,5)  bbox(tlbr)+cls
        tag_lens      = xs[4]     #(batch)
        ###############################################
        ct_inds       = xs[5]     #(batch,Max_target = 128)
        #tl_heatmaps_   = xs[6]     #(batch,80,128,128)
        #br_heatmaps_   = xs[7]     #(batch,80,128,128)
        #tag_masks_     = xs[8]     #(batch,Max_target = 128)
        #tl_regrs_      = xs[9]     #(batch,Max_target = 128,2)
        #br_regrs_      = xs[10]    #(batch,Max_target = 128,2)
        #ct_heatmaps_   = xs[11]    #(batch,80,128,128)
        #ct_regrs_      = xs[12]    #(batch,Max_target = 128,2)
        ###############################################
        
        num_imgs      = image.size(0)

        outs             = []
        #grouping_feats   = []
        region_feats     = []
        decode_inputs    = []
        grouping_list    = []
        gt_list          = []
        gt_labels        = []
        sampling_results = []
        #grouping_outs    = []
        region_outs      = []
        
        inter = self.pre(image)
        
        layers = zip(
            self.kps,         self.cnvs,
            self.tl_cnvs,     self.br_cnvs, 
            self.tl_heats,    self.br_heats,
            self.tl_regrs,    self.br_regrs,    
            self.regions,     self.groupings,
            self.ct_heats,    self.ct_regrs                                     ##############################
        )
        for ind, layer in enumerate(layers):
            kp_,        cnv_             = layer[0:2]
            tl_cnv_,     br_cnv_         = layer[2:4]
            tl_heat_,    br_heat_        = layer[4:6]
            tl_regr_,    br_regr_        = layer[6:8]
            region_,     grouping_       = layer[8:10]
            ############################################################
            ct_heat_,    ct_regr_        = layer[10:12]                         
            ############################################################

            kp = kp_(inter)
            cnv = cnv_(kp)
            
            tl_cnv = tl_cnv_(cnv)
            br_cnv = br_cnv_(cnv)
            
            region_feat    = region_(cnv)
            #grouping_feat  = grouping_(cnv)
            
            region_feats   += [region_feat]
            #grouping_feats += [grouping_feat]

            tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
            tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)
            ##############################################################
            ct_heat, ct_regr = ct_heat_(cnv), ct_regr_(cnv)                      
            ##############################################################
            
            decode_inputs += [tl_heat.clone().detach(), br_heat.clone().detach()]
            
            tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
            br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
            ########################################################################
            ct_regr = _tranpose_and_gather_feat(ct_regr, ct_inds)
            #######################################################################
            
            outs += [tl_heat, br_heat, tl_regr, br_regr, ct_heat, ct_regr] ################################################
            
            '''
            indicator = np.random.randint(0,100)
            file = 'debug_inputs/img_{}_{}.png'
            ori_img = image[0].cpu().numpy().transpose((1, 2, 0))
            cv2.imwrite(file.format(indicator,'ori'),ori_img)
            
            #cpr_img = ori_img[::4,::4,:].astype(np.uint8)
            cpr_img = cv2.resize(ori_img,(128,128))
            cv2.imwrite(file.format(indicator,'cpr'),cpr_img)
            
            
            file = 'debug_inputs/img_{}_{}.pt'
            with open(file.format(indicator,'tl_tags'),'w') as f:
                f.write(np.array2string(tl_inds.cpu().numpy()))
            with open(file.format(indicator,'br_tags'),'w') as f:
                f.write(np.array2string(br_inds.cpu().numpy()))
            with open(file.format(indicator,'gt_bboxes'),'w') as f:
                f.write(np.array2string(gt_detections.cpu().numpy()))
            with open(file.format(indicator,'tag_lens'),'w') as f:
                f.write(np.array2string(tag_lens.cpu().numpy()))            
            with open(file.format(indicator,'ct_tags'),'w') as f:
                f.write(np.array2string(ct_inds.cpu().numpy())) 

            _file = 'debug_inputs/img_{}_heatmap_{:02d}_{}.png'
            object_list = []
            for i in range(tag_lens):
                object_list.append(gt_detections[0,i,-1])
            for i in range(80):
                if i in object_list:
                    htmp = (tl_heatmaps_[0,i].cpu().numpy() * 255).astype(np.uint8)
                    htmp = cv2.applyColorMap(htmp, cv2.COLORMAP_JET)
                    cv2.imwrite(_file.format(indicator,i,'tl'),htmp) 
                else:
                    htmp = (tl_heatmaps_[0,i].cpu().numpy() * 255).astype(np.uint8)
                    if not np.all(htmp == 0):
                        print('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFalse for ',_file.format(indicator,i,'tl'))
                        fig, axs = plt.subplots()
                        axs.imshow(htmp,alpha = 1., interpolation = 'gaussian', cmap = plt.cm.jet)
                        plt.savefig(_file.format(indicator,i,'tl'))                               

            _file = 'debug_inputs/img_{}_heatmap_{:02d}_{}.png'
            object_list = []
            for i in range(tag_lens):
                object_list.append(gt_detections[0,i,-1])                
            for i in range(80):
                if i in object_list:
                    htmp = (br_heatmaps_[0,i].cpu().numpy() * 255).astype(np.uint8)
                    htmp = cv2.applyColorMap(htmp, cv2.COLORMAP_JET)
                    cv2.imwrite(_file.format(indicator,i,'br'),htmp) 
                else:
                    htmp = (br_heatmaps_[0,i].cpu().numpy() * 255).astype(np.uint8)
                    if not np.all(htmp == 0):
                        print('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFalse for ',_file.format(indicator,i,'tl'))
                        fig, axs = plt.subplots()
                        axs.imshow(htmp,alpha = 1., interpolation = 'gaussian', cmap = plt.cm.jet)
                        plt.savefig(_file.format(indicator,i,'tl'))                          
            with open(file.format(indicator,'tag_masks'),'w') as f:
                f.write(np.array2string(tag_masks_.cpu().numpy()))    
            with open(file.format(indicator,'tl_regrs'),'w') as f:
                f.write(np.array2string(tl_regrs_.cpu().numpy()))
            with open(file.format(indicator,'br_regrs'),'w') as f:
                f.write(np.array2string(br_regrs_.cpu().numpy()))

            _file = 'debug_inputs/img_{}_heatmap_{:02d}_{}.png'
            object_list = []
            for i in range(tag_lens):
                object_list.append(gt_detections[0,i,-1])                     
            for i in range(80):
                if i in object_list:
                    htmp = (ct_heatmaps_[0,i].cpu().numpy() * 255).astype(np.uint8)
                    htmp = cv2.applyColorMap(htmp, cv2.COLORMAP_JET)
                    cv2.imwrite(_file.format(indicator,i,'ct'),htmp)  
                else:
                    htmp = (ct_heatmaps_[0,i].cpu().numpy() * 255).astype(np.uint8)
                    if not np.all(htmp == 0):
                        print('FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFalse for ',_file.format(indicator,i,'tl'))
                        fig, axs = plt.subplots()
                        axs.imshow(htmp,alpha = 1., interpolation = 'gaussian', cmap = plt.cm.jet)
                        plt.savefig(_file.format(indicator,i,'tl'))                            
            with open(file.format(indicator,'ct_regrs'),'w') as f:
                f.write(np.array2string(ct_regrs_.cpu().numpy()))
            '''
            
            if ind == self.nstack - 1:
                ht_boxes, tlbr_inds, tlbr_scores, tl_clses = self._generate_bboxes(decode_inputs[-2:])  # decode_inputs[-1] : (batch,80,128,128)  ht_boxes: (batch,K*K,5) tlbr_inds:(batch*2,K) tlbr_scores:(batch*2,K,K) tl_clses:(batch,K,K)
                '''
                for i in range(num_imgs):
                    gt_box     = gt_detections[i][:tag_lens[i]][:,:4] 
                    ht_box     = ht_boxes[i]
                    score_inds = ht_box[:,4] > 0 
                    ht_box     = ht_box[score_inds, :4]
                    
                    if ht_box.size(0) == 0:
                        grouping_list += [gt_box] 
                    else:
                        grouping_list += [ht_box]
                        
                    gt_list   += [gt_box]
                    gt_labels += [(gt_detections[i,:tag_lens[i], -1]+ 1).long()]
                
                bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
                bbox_sampler  = build_sampler(self.train_cfg.rcnn.sampler, context=self)
                
                gt_list_ignore = [None for _ in range(num_imgs)]
                
                for i in range(num_imgs):
                    assign_result = bbox_assigner.assign(grouping_list[i], gt_list[i], gt_list_ignore[i], gt_labels[i])
                    sampling_result = bbox_sampler.sample(self.categories, assign_result, grouping_list[i], gt_list[i], gt_labels[i])
                    sampling_results.append(sampling_result)
                
                grouping_rois = bbox2roi([res.bboxes for res in sampling_results]) 
                box_targets   = self.bbox_head.get_target(sampling_results, gt_list, gt_labels, self.train_cfg.rcnn)
                roi_labels = box_targets[0]
                gt_label_inds = roi_labels > self.categories
                roi_labels[gt_label_inds] -= self.categories
                grouping_inds = roi_labels > 0
                grouping_labels = grouping_rois.new_full((grouping_rois.size(0), 1, 1, 1), 0, dtype=torch.float).cuda()
                grouping_labels[grouping_inds] = 1
                region_labels = grouping_rois.new_full((grouping_rois.size(0), self.categories+1), 0, dtype=torch.float).cuda()
                region_labels = region_labels.scatter_(1, roi_labels.unsqueeze(-1), 1)
                region_labels = region_labels[:,1:].unsqueeze(-1).unsqueeze(-1)
                
                grouping_roi_feats = self.grouping_roi_extractor(grouping_feats, grouping_rois)
                
                for grouping_reduce, grouping_roi_feat in zip(self.grouping_reduces, grouping_roi_feats):
                    grouping_outs += [_sigmoid(grouping_reduce(grouping_roi_feat))]
                 
                grouping_scores = grouping_outs[-1][:,0,0,0].clone().detach()
                grouping_scores[gt_label_inds] = 1
                select_inds = grouping_scores >= self.gr_threshold
                region_rois = grouping_rois[select_inds].contiguous()
                region_labels = region_labels[select_inds]
                    
                region_roi_feats = self.region_roi_extractor(region_feats, region_rois)
                for region_reduce, region_roi_feat in zip(self.region_reduces, region_roi_feats):
                     region_outs += [_sigmoid(region_reduce(region_roi_feat))]
                    
                outs += [grouping_outs, grouping_labels, region_outs, region_labels]
                '''
                ############################################################################################################
                for i in range(num_imgs):
                    gt_box     = gt_detections[i][:tag_lens[i]][:,:4]           #(gt_target_number,4)
                    ht_box     = ht_boxes[i]                                    #(K*K,5)
                    score_inds = ht_box[:,4] > 0 
                    ht_box     = ht_box[score_inds, :4]                         #(pred_target_number,4)
                    
                    ht_box_cls = tl_clses[i].contiguous().view(-1)  #(K*K)
                    ht_box_cls = ht_box_cls[score_inds] #(number_of_pos_pred,)
                    ht_box = center_filtering_train(ct_heat[i].clone().detach(),ht_box,ht_box_cls) # returned ht_box : (number_of_pos_pos_pred,4) 
                    
                    if ht_box.size(0) == 0:
                        grouping_list += [gt_box]                               
                    else:
                        grouping_list += [ht_box]                               #(batch,target_number_per_batch,4)
                        
                    gt_list   += [gt_box]                                       #(batch,gt_target_number_per_batch,4)
                    gt_labels += [(gt_detections[i,:tag_lens[i], -1]+ 1).long()]#(batch,gt_target_number_per_batch)    cls starts from 1 to 80
                
                bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)    #assigner=dict(type='MaxIoUAssigner',pos_iou_thr=0.7,neg_iou_thr=0.7,min_pos_iou=0.5,ignore_iof_thr=-1)
                bbox_sampler  = build_sampler(self.train_cfg.rcnn.sampler, context=self) #sampler=dict(type='RandomSampler',num=256,pos_fraction=0.5,neg_pos_ub=-1,add_gt_as_proposals=True) ***********now use 30 x 30
                
                gt_list_ignore = [None for _ in range(num_imgs)] 

                for i in range(num_imgs):
                    assign_result = bbox_assigner.assign(grouping_list[i], gt_list[i], gt_list_ignore[i], gt_labels[i]) #This method assign a gt bbox to every bbox (proposal/anchor), each bbox will be assigned with -1, 0, or a positive number 
                    '''
                    Args:
                    bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
                    gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
                    gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                    labelled as `ignored`, e.g., crowd boxes in COCO.
                    gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).
                    
                    Returns:
                    obj:`AssignResult`: The assign result.
                    '''
                    sampling_result = bbox_sampler.sample(self.categories, assign_result, grouping_list[i], gt_list[i], gt_labels[i]) # Sample positive and negative bboxes.
                    '''
                    Args:
                    assign_result (:obj:`AssignResult`): Bbox assigning results.
                    bboxes (Tensor): Boxes to be sampled from.
                    gt_bboxes (Tensor): Ground truth bboxes.
                    gt_labels (Tensor, optional): Class labels of ground truth bboxes.
                    
                    Returns:
                    obj:`SamplingResult`: Sampling result.
                    '''
                    sampling_results.append(sampling_result)    #(batch)
                
                region_rois = bbox2roi([res.bboxes for res in sampling_results])  # Convert a list of bboxes to roi format.
                '''
                Args:
                bbox_list (list[Tensor]): a list of bboxes corresponding to a batch of images.
                
                Returns:
                Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
                '''
                box_targets   = self.bbox_head.get_target(sampling_results, gt_list, gt_labels, self.train_cfg.rcnn)  #dict(type='SharedFCBBoxHead',num_fcs=1,in_channels=1,fc_out_channels=1,roi_feat_size=1,num_classes=1,target_means=[0., 0., 0., 0.],target_stds=[0.1, 0.1, 0.2, 0.2],reg_class_agnostic=False, with_reg = False)
                '''
                Returns: 
                labels, label_weights, bbox_targets, bbox_weights | Targets of boxes and class prediction.
                '''
                roi_labels = box_targets[0] # roi_labels contain 0 corresponding to negative examples 1-80 corresponding to 80 classes, region_rois.size(0) is number of targets (sampling results positive classes + negative classes)
                gt_label_inds = roi_labels > self.categories
                roi_labels[gt_label_inds] -= self.categories   # roi_labels <= 80; roi_labels could be larger than 80
                region_labels = region_rois.new_full((region_rois.size(0), self.categories+1), 0, dtype=torch.float).cuda() # contains 80 classes + 1 negative class
                region_labels = region_labels.scatter_(1, roi_labels.unsqueeze(-1), 1)
                region_labels = region_labels[:,1:].unsqueeze(-1).unsqueeze(-1)   #(region_rois.size(0), 80,1,1)
                
                '''
                indicator = region_labels.get_device()
                file = 'debug_train/img_{}.png'
                ori_img = image[0].cpu().numpy().transpose((1, 2, 0))
                cpr_img = cv2.resize(ori_img,(128,128))
                cv2.imwrite(file.format(indicator),cpr_img)               
                file = 'debug_train/roi_labels_{}.pt'
                with open(file.format(indicator),'w') as f:
                    f.write(np.array2string(roi_labels.cpu().numpy(),threshold=np.inf))
                file = 'debug_train/region_labels_{}.pt'
                with open(file.format(indicator),'w') as f:
                    f.write(np.array2string(region_labels.cpu().numpy(),threshold=np.inf))
                file = 'debug_train/bbox_targets_{}.pt'
                with open(file.format(indicator),'w') as f:
                    f.write(np.array2string(box_targets[2].cpu().numpy(),threshold=np.inf))  
                pos_proposals = [res.pos_bboxes for res in sampling_results]
                neg_proposals = [res.neg_bboxes for res in sampling_results]
                pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
                pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
                file = 'debug_train/pos_proposals_{}.pt'                                            
                with open(file.format(indicator),'w') as f:
                    f.write(np.array2string(np.array(pos_proposals),threshold=np.inf))  
                file = 'debug_train/neg_proposals_{}.pt'
                with open(file.format(indicator),'w') as f:
                    f.write(np.array2string(np.array(neg_proposals),threshold=np.inf))                  
                file = 'debug_train/pos_gt_bboxes_{}.pt'
                with open(file.format(indicator),'w') as f:
                    f.write(np.array2string(np.array(pos_gt_bboxes),threshold=np.inf))  
                file = 'debug_train/pos_gt_labels_{}.pt'
                with open(file.format(indicator),'w') as f:
                    f.write(np.array2string(np.array(pos_gt_labels),threshold=np.inf))   
                '''
                '''
                pos_proposals = [res.pos_bboxes for res in sampling_results]
                neg_proposals = [res.neg_bboxes for res in sampling_results]
                print('_____pos_proposals.shape_____',pos_proposals[0].shape)
                print('____neg_proposals.shape____',neg_proposals[0].shape)
                '''
                
                
                region_roi_feats = self.region_roi_extractor(region_feats, region_rois) #dict(type='SingleRoIExtractor',roi_layer=dict(type='RoIAlign', out_size=7, sample_num=self._configs["roi_sample_num"]),out_channels=256,featmap_strides=[1] if self._configs["featmap_strides"] == 1 else [1,1])
                '''
                Extract RoI features from a single level feature map.
                '''
                for region_reduce, region_roi_feat in zip(self.region_reduces, region_roi_feats):
                     region_outs += [_sigmoid(region_reduce(region_roi_feat))]    # region_outs[0].shape  (region_rois.size(0),80,1,1)
                
                outs += [region_outs, region_labels]  
                ##########################################################################################################################                
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs

    def _test(self, *xs, **kwargs):
        image     = xs[0]
        no_flip   = kwargs.pop('no_flip')  # flip augmentation default: false (set by command arguments)
        image_idx = kwargs['image_idx']  # current image index
        kwargs.pop('image_idx')
        
        num_imgs = image.size(0)
        
        inter    = self.pre(image)

        outs            = []
        region_feats    = []
        #grouping_feats  = []
        decode_inputs   = []
        region_list   = []
        score_inds_list = []

        layers = zip(
            self.kps,           self.cnvs,
            self.tl_cnvs,       self.br_cnvs, 
            self.tl_heats,      self.br_heats,     
            self.tl_regrs,      self.br_regrs,       
            self.regions,       self.groupings,
            self.ct_heats,      self.ct_regrs                                     ##############################
        )
        
        for ind, layer in enumerate(layers):
            kp_,           cnv_      = layer[0:2]
            tl_cnv_,       br_cnv_   = layer[2:4]
            tl_heat_,      br_heat_  = layer[4:6]
            tl_regr_,      br_regr_  = layer[6:8]
            region_,       grouping_ = layer[8:10]
            ############################################################
            ct_heat_,      ct_regr_  = layer[10:12]                         
            ############################################################            

            kp = kp_(inter)
            cnv = cnv_(kp)
            
            if ind == self.nstack - 1:
                tl_cnv = tl_cnv_(cnv)
                br_cnv = br_cnv_(cnv)
                
                region_feat    = region_(cnv)
                #grouping_feat  = grouping_(cnv)
                
                region_feats   += [region_feat]
                #grouping_feats += [grouping_feat]

                tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
                tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)
                ##############################################################
                ct_heat, ct_regr = ct_heat_(cnv), ct_regr_(cnv)                      
                ##############################################################  
                
                decode_inputs += [tl_heat.clone().detach(), br_heat.clone().detach()]

                outs += [tl_regr, br_regr]
                
                ht_boxes, tlbr_inds, tlbr_scores, tl_clses = self._generate_bboxes(decode_inputs[-2:]) # decode_inputs[-1] : (batch,80,128,128)  ht_boxes: (batch,K*K,5) tlbr_inds:(batch*2,K) tlbr_scores:(batch*2,K,K) tl_clses:(batch,K,K)
                '''
                all_groupings = ht_boxes[:,:, -1].new_full(ht_boxes[:,:, -1].size(), 0, dtype=torch.float)  #(batch,K*K)
                
                for i in range(num_imgs):
                    ht_box      = ht_boxes[i]                 #(K*K,5)
                    score_inds  = ht_box[:,4] > 0
                    ht_box      = ht_box[score_inds, :4]      #(number_of_pos_pred,4)
                    
                    grouping_list  += [ht_box]                #(batch,number_of_pos_pred,4)
                    score_inds_list+= [score_inds.unsqueeze(0)]     #(batch,1,K*K)
                        
                grouping_rois = _htbox2roi_test(grouping_list)    # Convert a list of bboxes to roi format.
                ###Args:
                ###bbox_list (list[Tensor]): a list of bboxes corresponding to a batch of images.
                
                ###Returns:
                ###Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
                grouping_roi_feats = self.grouping_roi_extractor(grouping_feats, grouping_rois.float()) #dict(type='SingleRoIExtractor',roi_layer=dict(type='RoIAlign', out_size=7, sample_num= self._configs["roi_sample_num"]),out_channels=32,featmap_strides=[1] if self._configs["featmap_strides"] == 1 else [1,1])
                
                grouping_scores = self.grouping_reduces[-1](grouping_roi_feats[-1])
                grouping_scores = _sigmoid(grouping_scores) #(grouping_rois.size[0] == all positive number of prediction,1,1,1)
                
                grouping_inds = grouping_scores[:,0,0,0] >= self.gr_threshold  #(grouping_rois.size[0] == all positive number of prediction,)
                
                if grouping_inds.float().sum() > 0:                                     #select those grouping_rois >= threshold 
                    region_rois = grouping_rois[grouping_inds].contiguous().float()
                else:                                                                   #handle the case that there is no surviving grouping_rois
                    region_rois = grouping_rois
                
                region_roi_feats = self.region_roi_extractor(region_feats, region_rois)  #dict(type='SingleRoIExtractor',roi_layer=dict(type='RoIAlign', out_size=7, sample_num=self._configs["roi_sample_num"]),out_channels=256,featmap_strides=[1] if self._configs["featmap_strides"] == 1 else [1,1])
                region_scores    = self.region_reduces[-1](region_roi_feats[-1])         
                region_scores    = _sigmoid(region_scores)                               #(region_rois.size(0),80,1,1)
                
                if grouping_inds.float().sum() > 1:     # handle when more than one postive predition, modify score in ht_box
                     _filter_bboxes(ht_boxes, tl_clses, region_scores, grouping_scores, self.gr_threshold)   
                        
                if no_flip:
                    all_groupings[score_inds_list[0]] = grouping_scores[:,0,0,0]  # testing input image batch size = 1 if no_flip, fill all_grouping (batch = 1,K*K)
                else:
                    all_groupings[torch.cat((score_inds_list[0], score_inds_list[1]), 0)] = grouping_scores[:,0,0,0]  # testing input image batch size = 2 if not no_flip, fill all_grouping (batch = 2,K*K)
                
                outs += [ht_boxes, all_groupings, tlbr_inds, tlbr_scores, tl_clses, self.gr_threshold]
                '''
                for i in range(num_imgs):
                    ht_box     = ht_boxes[i]  #(K*K,5)
                    score_inds = ht_box[:,4] > 0 
                    ht_box     = ht_box[score_inds, :]   #(number_of_pos_pred,5)
                    
                    ht_box_cls = tl_clses[i].contiguous().view(-1)  #(K*K)
                    ht_box_cls = ht_box_cls[score_inds] #(number_of_pos_pred,)
                    
                    ht_box,ht_box_new_score = center_filtering_test(ct_heat[i].clone().detach(),ht_box,ht_box_cls) # ht_box : (number_of_pos_pos_pred,4)  ht_box_new_score : (number_of_pos_pred,)
                    
                    region_list  += [ht_box] #(batch,number_of_pos_pos_pred,4)
                    ht_boxes[i,score_inds,4] = ht_box_new_score #(number_of_pos_pred,)
                    
                region_rois = _htbox2roi_test(region_list) # Convert a list of bboxes to roi format. n = all positive examples over batches
                ###Args:
                ###bbox_list (list[Tensor]): a list of bboxes corresponding to a batch of images.
                
                ###Returns:
                ###Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]      
                region_roi_feats = self.region_roi_extractor(region_feats, region_rois.float()) #dict(type='SingleRoIExtractor',roi_layer=dict(type='RoIAlign', out_size=7, sample_num=self._configs["roi_sample_num"]),out_channels=256,featmap_strides=[1] if self._configs["featmap_strides"] == 1 else [1,1])
                region_scores    = self.region_reduces[-1](region_roi_feats[-1])
                region_scores    = _sigmoid(region_scores)   #(region_rois.size(0) = n,80,1,1)
                
                _filter_bboxes(ht_boxes, tl_clses, region_scores)
                
                outs += [ht_boxes, tlbr_inds, tlbr_scores, tl_clses]                
                
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
                
        return self._decode(*outs[-6:], **kwargs)
    
    def forward(self, *xs, **kwargs):                       
        if len(xs) > 1:
            return self._train(*xs, **kwargs)          
        return self._test(*xs, **kwargs)

class AELoss52(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=_neg_loss, ct_heat_weight=2):
        super(AELoss52, self).__init__()

        self.pull_weight   = pull_weight
        self.push_weight   = push_weight
        self.regr_weight   = regr_weight
        self.focal_loss    = focal_loss
        self.ae_loss       = _ae_loss
        self.regr_loss     = _regr_loss
        self._regr_l1_loss = _regr_l1_loss
        self.ct_heat_weight = ct_heat_weight

    def forward(self, outs, targets):
        region_labels   = outs.pop(-1)   #(number_of_rois,80,1,1)
        region_outs     = outs.pop(-1)   #(number_of_rois,80,1,1)
        #grouping_labels = outs.pop(-1)
        #grouping_outs   = outs.pop(-1)
        
        stride = 6

        tl_heats = outs[0::stride] #(batch,80,128,128)
        br_heats = outs[1::stride] #(batch,80,128,128)
        tl_regrs = outs[2::stride] #(batch,Max_target = 128,2)
        br_regrs = outs[3::stride] #(batch,Max_target = 128,2)
        #####################################################
        ct_heats = outs[4::stride] #(batch,80,128,128)
        ct_regrs = outs[5::stride] #(batch,Max_target = 128,2)
        #####################################################        
        
        
        gt_tl_heat = targets[0] #(batch,80,128,128)
        gt_br_heat = targets[1] #(batch,80,128,128)
        gt_mask    = targets[2] #(batch,Max_target = 128)
        gt_tl_regr = targets[3] #(batch,Max_target = 128,2)
        gt_br_regr = targets[4] #(batch,Max_target = 128,2)
        #####################################################
        gt_ct_heat = targets[5] #(batch,80,128,128)
        gt_ct_regr = targets[6] #(batch,Max_target = 128,2)
        ####################################################      
        
        # keypoints loss
        focal_loss = 0

        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]
        #####################################################
        ct_heats = [_sigmoid(c) for c in ct_heats]
        #####################################################

        focal_loss += self.focal_loss(tl_heats, gt_tl_heat)
        focal_loss += self.focal_loss(br_heats, gt_br_heat)
        #####################################################
        focal_loss += self.focal_loss(ct_heats, gt_ct_heat) * self.ct_heat_weight
        ######################################################
        
        # grouping loss
        #grouping_loss = 0
        #grouping_loss+= self.focal_loss(grouping_outs, grouping_labels)
        
        # region loss
        region_loss = 0
        region_loss+= self.focal_loss(region_outs, region_labels)

        regr_loss = 0
        for tl_regr, br_regr, ct_regr in zip(tl_regrs, br_regrs, ct_regrs):
            regr_loss += self.regr_loss(tl_regr, gt_tl_regr, gt_mask)
            regr_loss += self.regr_loss(br_regr, gt_br_regr, gt_mask)
            ###############################################################
            regr_loss += self.regr_loss(ct_regr, gt_ct_regr, gt_mask)
            ###############################################################
        regr_loss = self.regr_weight * regr_loss
        
        #loss = (focal_loss + grouping_loss + region_loss + regr_loss) / len(tl_heats)
        loss = (focal_loss + region_loss + regr_loss) / len(tl_heats)   # len(tl_heats) == 1
        
        return loss.unsqueeze(0), (focal_loss / len(tl_heats)).unsqueeze(0), \
                          (region_loss / len(tl_heats)).unsqueeze(0), (regr_loss / len(tl_heats)).unsqueeze(0)
