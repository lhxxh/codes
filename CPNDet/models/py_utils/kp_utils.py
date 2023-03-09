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
import torch.nn as nn
import torch.nn.functional as F
from .utils import convolution, residual
import sys
import numpy as np

class MergeUp(nn.Module):
    def forward(self, up1, up2):
        return up1 + up2

def make_merge_layer(dim):
    return MergeUp()

def make_tl_layer(dim):
    return None

def make_br_layer(dim):
    return None

def make_region_layer(dim):
    return None

def make_pool_layer(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)

def make_unpool_layer(dim):
    return nn.Upsample(scale_factor=2)

def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )

def make_inter_layer(dim):
    return residual(3, dim, dim)

def make_cnv_layer(inp_dim, out_dim):
    return convolution(3, inp_dim, out_dim)

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _htbox2roi(detections, proposals, roi_lables, tag_lens):
    for i in range(len(tag_lens)):
        if tag_lens[i] != 0:
           proposals[i] = torch.cat((detections[i, :tag_lens[i],:5], proposals[i]), dim = 0) \
                     if proposals[i].size(0) > 0 else detections[i, :tag_lens[i],:5]
        
           gt_labels     = roi_lables[i].new_full((tag_lens[i], ), 1, dtype=torch.float)
           roi_lables[i] = torch.cat((gt_labels, roi_lables[i]), dim = 0) \
                      if roi_lables[i].size(0) > 0 else gt_labels
           #unique 
           overlaps = bbox_overlaps(proposals[i][:,:4], proposals[i][:,:4])
           new_proposal_inds = overlaps.new_full((overlaps.size(1), ), 0, dtype=torch.uint8)
           max_overlaps, argmax_overlaps = overlaps.max(dim=0)
           new_proposal_inds[argmax_overlaps] = 1
           proposals[i] = proposals[i][new_proposal_inds]
           roi_lables[i] = roi_lables[i][new_proposal_inds]
            
    rois_list = []
    for img_id, bboxes in enumerate(proposals):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :5]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 6))
        rois_list += [rois]
    rois = torch.cat(rois_list, 0)
    return rois, roi_lables


def _htbox2roi_test(proposals):
    rois_list = []
    for img_id, bboxes in enumerate(proposals):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list += [rois]
    rois = torch.cat(rois_list, 0)
    return rois

def _nms(heat, kernel=1):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)      # select the top k from each category per batch

    topk_inds = topk_inds % (height * width)                                 # discard cls info
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)       # select the top k per batch
    topk_clses = (topk_ind / K).int()
    topk_inds  = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)              # get index according to original shape
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

'''
def _filter_bboxes(ht_boxes, tl_clses, region_scores, grouping_scores, gr_threshold):
    grouping_scores = grouping_scores[:,0,0,0]   #(grouping_rois.size[0] == all positive number of prediction,)
    batch  = tl_clses.size(0)   
    region_scores = region_scores.squeeze()   #(region_rois.size(0) = positive_postive_prediction,80)
    
    ht_scores = ht_boxes[:,:,-1].unsqueeze(-1)  #(batch,K*K,1)
    clses = tl_clses.contiguous().view(batch, -1, 1).float() #(batch,K*K,1)
    
    ht_score_cls = torch.cat((ht_scores, clses), dim = -1)   #(batch,K*K,2)
    pos_inds = ht_boxes[:,:, -1] > 0   #(batch,K*K)
    pos_ht_score_cls = ht_score_cls[pos_inds]  #(postive_prediction,2)
    
    pos_grouping_score_inds = grouping_scores >= gr_threshold  #(grouping_rois.size[0] == all positive number of prediction,)
    ppos_ht_score_cls = pos_ht_score_cls[pos_grouping_score_inds] #(positive_postive_prediction,2)
    
    specific_rscores = region_scores.gather(1, ppos_ht_score_cls[:,1].long().unsqueeze(-1)).squeeze()  #(positive_postive_prediction,1) extract prediction rscore from predited class
    #specific_rscores[specific_rscores<0.1] = 0
    
    ppos_ht_score_cls[:,0] = (ppos_ht_score_cls[:,0] + 0.5)* (specific_rscores + 0.5) - 0.25 #(positive_postive_prediction,2)  modify score in ppos_ht_score_cls
    
    pos_ht_score_cls[pos_grouping_score_inds] = ppos_ht_score_cls #(postive_prediction,2) modify score in pos_ht_score_cls
    
    ht_scores[pos_inds] = pos_ht_score_cls[:,:1]  #(batch,K*K,1)


def _decode(
    tl_regr, br_regr, ht_boxes, all_groupings, tlbr_inds, tlbr_scores, tl_clses, gr_threshold,
    K=100, kernel=1, ae_threshold=1, num_dets=1000, ratios = 0
):
    batch = tl_regr.size(0)

    bboxes = ht_boxes[:,:,:4]     #(batch,K*K,4)
    
    tl_ys = bboxes.view(batch, K, K, -1)[:,:,:,1]  #(batch,K,K)
    tl_xs = bboxes.view(batch, K, K, -1)[:,:,:,0]  #(batch,K,K)
    br_ys = bboxes.view(batch, K, K, -1)[:,:,:,3]  #(batch,K,K)
    br_xs = bboxes.view(batch, K, K, -1)[:,:,:,2]  #(batch,K,K)
    
    scores = ht_boxes[:,:,-1].unsqueeze(-1) #(batch,K*K,1)
    
    tl_inds = tlbr_inds[:batch,:] #(batch,K)
    br_inds = tlbr_inds[batch:,:] #(batch,K)
    
    tl_scores = tlbr_scores[:batch,...] #(batch,K,K)
    br_scores = tlbr_scores[batch:,...] #(batch,K,K)

    if tl_regr is not None and br_regr is not None:
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds) #(batch,K,2)
        tl_regr = tl_regr.view(batch, K, 1, 2)                #(batch,k,1,2)   tl_xs broadcast from left to right
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds) #(batch,K,2)
        br_regr = br_regr.view(batch, 1, K, 2)                #(batch,1,K,2)   br_regr broadcast from top to bottom

        tl_xs = tl_xs + tl_regr[..., 0]   #(batch,K,K)
        tl_ys = tl_ys + tl_regr[..., 1]   #(batch,K,K)
        br_xs = br_xs + br_regr[..., 0]   #(batch,K,K)
        br_ys = br_ys + br_regr[..., 1]   #(batch,K,K)

    # all possible boxes based on top k corners (ignoring class)
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3) #(batch,K,K,4)
    
    all_groupings = all_groupings.view(batch, K, K)   #(batch = 1 if not non-flip or 2 if non-flip,K,K)
    dist_inds = (all_groupings < gr_threshold) #(batch,K,K)

    scores -= dist_inds.view(batch, -1, 1).float() #(batch,K*K,1) # eliminate those groupings whose score less than threshold 

    scores = scores.view(batch, -1)  #(batch,K*K)
    scores, inds = torch.topk(scores, num_dets) #(batch,num_dets)
    scores = scores.unsqueeze(2)  #(batch,num_dets,1)

    bboxes = bboxes.view(batch, -1, 4)  #(batch,K*K,4)
    bboxes = _gather_feat(bboxes, inds)  #(batch,num_dets,4)
    
    clses  = tl_clses.contiguous().view(batch, -1, 1)  #(batch,K*K,1)
    clses  = _gather_feat(clses, inds).float()  #(batch,num_dets,1)

    tl_scores = tl_scores.contiguous().view(batch, -1, 1) #(batch,K*K,1)
    tl_scores = _gather_feat(tl_scores, inds).float()     #(batch,num_dets,1)
    br_scores = br_scores.contiguous().view(batch, -1, 1) #(batch,K*K,1)
    br_scores = _gather_feat(br_scores, inds).float()     #(batch,num_dets,1)
    
    detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)
    
    return detections
'''

def _filter_bboxes(ht_boxes, tl_clses, region_scores): 
    batch  = tl_clses.size(0) 
    region_scores = region_scores.squeeze() #(region_rois.size(0) = n,80)
    ################################
    region_scores = region_scores.unsqueeze(1) # now class changed to 1
    ###############################
    
    ht_scores = ht_boxes[:,:,-1].unsqueeze(-1) #(batch,K*K,1)
    clses = tl_clses.contiguous().view(batch, -1, 1).float() #(batch,K*K,1)
    
    ht_score_cls = torch.cat((ht_scores, clses), dim = -1) #(batch,K*K,2)
    pos_inds = ht_boxes[:,:, -1] > 0  #(batch,K*K)
    pos_ht_score_cls = ht_score_cls[pos_inds] #(postive_prediction,2)
    
    specific_rscores = region_scores.gather(1, pos_ht_score_cls[:,1].long().unsqueeze(-1)).squeeze() #(postive_prediction,1)
    
    pos_ht_score_cls[:,0] = (pos_ht_score_cls[:,0] + 0.5)* (specific_rscores + 0.5) - 0.25  #(postive_prediction,)
    
    ht_scores[pos_inds] = pos_ht_score_cls[:,:1] #(postive_prediction,)
    

def _decode(
    tl_regr, br_regr, ht_boxes, tlbr_inds, tlbr_scores, tl_clses,
    K=100, kernel=1, ae_threshold=1, num_dets=1000, ratios = 0
):
    batch = tl_regr.size(0)

    bboxes = ht_boxes[:,:,:4]  #(batch,K*K,4)
    
    tl_ys = bboxes.view(batch, K, K, -1)[:,:,:,1] #(batch,K,K)
    tl_xs = bboxes.view(batch, K, K, -1)[:,:,:,0] #(batch,K,K)
    br_ys = bboxes.view(batch, K, K, -1)[:,:,:,3] #(batch,K,K)
    br_xs = bboxes.view(batch, K, K, -1)[:,:,:,2] #(batch,K,K)
    
    scores = ht_boxes[:,:,-1].unsqueeze(-1) #(batch,K*K,1)
    
    tl_inds = tlbr_inds[:batch,:] #(batch,K)
    br_inds = tlbr_inds[batch:,:] #(batch,K)
    
    tl_scores = tlbr_scores[:batch,...]#(batch,K,K)
    br_scores = tlbr_scores[batch:,...]#(batch,K,K)

    if tl_regr is not None and br_regr is not None:
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds) #(batch,K,2)
        tl_regr = tl_regr.view(batch, K, 1, 2)  #(batch,k,1,2)   tl_regr broadcast from left to right
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds) #(batch,K,2)
        br_regr = br_regr.view(batch, 1, K, 2)  #(batch,1,K,2)   br_regr broadcast from top to bottom

        tl_xs = tl_xs + tl_regr[..., 0] #(batch,K,K)
        tl_ys = tl_ys + tl_regr[..., 1] #(batch,K,K)
        br_xs = br_xs + br_regr[..., 0] #(batch,K,K)
        br_ys = br_ys + br_regr[..., 1] #(batch,K,K)

    # all possible boxes based on top k corners (ignoring class)
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3) #(batch,K,K,4)

    scores = scores.view(batch, -1)  #(batch,K*K)
    scores, inds = torch.topk(scores, num_dets) #(batch,num_dets)
    scores = scores.unsqueeze(2) #(batch,num_dets,1)

    bboxes = bboxes.view(batch, -1, 4)  #(batch,K*K,4)
    bboxes = _gather_feat(bboxes, inds)  #(batch,num_dets,4)
    
    clses  = tl_clses.contiguous().view(batch, -1, 1)  #(batch,K*K,1)
    clses  = _gather_feat(clses, inds).float()   #(batch,num_dets,1)

    tl_scores = tl_scores.contiguous().view(batch, -1, 1)  #(batch,K*K,1)
    tl_scores = _gather_feat(tl_scores, inds).float()  #(batch,num_dets,1)
    br_scores = br_scores.contiguous().view(batch, -1, 1)  #(batch,K*K,1)
    br_scores = _gather_feat(br_scores, inds).float()  #(batch,num_dets,1)
    
    detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)
    
    return detections

def _generate_bboxes(
    decode_inputs, K=70, kernel=3
):
    tl_heat   = decode_inputs[0]
    br_heat   = decode_inputs[1]
    
    batch, cat, height, width = tl_heat.size()

    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)

    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)   # all size with (batch,K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)   # all size with (batch,K)

    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)

    # all possible boxes based on top k corners (ignoring class)
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)       # (batch,K,K,4)
    
    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
    br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
    scores   = (tl_scores + br_scores) / 2                          # (batch,K,K)

    # reject boxes based on classes
    tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
    br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
    cls_inds = (tl_clses != br_clses)

    # reject boxes based on widths and heights
    width_inds   = (br_xs < tl_xs)
    height_inds  = (br_ys < tl_ys)
    
    neg_inds = cls_inds | width_inds | height_inds 
    scores -= neg_inds.float()

    #scores[cls_inds]    = -1
    #scores[width_inds]   = -1
    #scores[height_inds]  = -1

    scores = scores.view(batch, -1)                                #(batch,K*K)
    scores = scores.unsqueeze(2)                                   #(batch,K*K,1)

    bboxes = bboxes.view(batch, -1, 4)                             #(batch,K*K,4)
    detections = torch.cat([bboxes, scores], dim=2)                #(batch,K*K,5)
    
    return detections, torch.cat([tl_inds, br_inds], dim=0), torch.cat([tl_scores, br_scores], dim=0), tl_clses
    
def center_filtering_train (ct_heat,ht_box,ht_box_cls,kernel=3,K=70): # skip regressor part
    ct_heat = ct_heat.unsqueeze(0)    #(batch = 1,80,128,128)
    ct_heat = torch.sigmoid(ct_heat)   
    ct_heat = _nms(ct_heat, kernel=kernel) #(batch = 1,80,128,128)
    ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = _topk(ct_heat, K=K)   # all size with (batch = 1,K)
    ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = ct_scores[0], ct_inds[0], ct_clses[0], ct_ys[0], ct_xs[0]   # all size with (K,)    

    n = 3
    ct_box = ht_box.new_full(ht_box[:,:4].shape,0)  #(number_of_pos_pred,4)
    ct_box[:,0] = ((n+1)*ht_box[:,0] + (n-1)*ht_box[:,2])/(2*n) #ctlx (number_of_pos_pred,)
    ct_box[:,1] = ((n+1)*ht_box[:,1] + (n-1)*ht_box[:,3])/(2*n) #ctly (number_of_pos_pred,)
    ct_box[:,2] = ((n-1)*ht_box[:,0] + (n+1)*ht_box[:,2])/(2*n) #cbrx (number_of_pos_pred,)
    ct_box[:,3] = ((n-1)*ht_box[:,1] + (n+1)*ht_box[:,3])/(2*n) #cbry (number_of_pos_pred,)
    ct_box_cls = ht_box_cls #(number_of_pos_pred,)    
    
    proposal_ind = ht_box.new_full(ht_box[:,0].shape,False,dtype = torch.uint8) #(number_of_pos_pred,)
    
    for i in range(K):
        cty,ctx = ct_ys[i], ct_xs[i] # all size with (1,)
        ct_cls = ct_clses[i] # (1,)
        
        true = torch.as_tensor(True,device = ct_box.get_device())
        false = torch.as_tensor(False,device = ct_box.get_device())
        ct_tpk_filter = torch.where(ct_box_cls == ct_cls,true,false) & torch.where(ct_box[:,0]<=ctx,true,false) & torch.where(ct_box[:,1]<=cty,true,false) & torch.where(ct_box[:,2]>=ctx,true,false) & torch.where(ct_box[:,3]>=cty,true,false)#(number_of_pos_pred,)
        proposal_ind |= ct_tpk_filter #(number_of_pos_pred,)
        
    ret_box = ct_box[proposal_ind,:]  # (number_of_pos_pos_pred,4)
    
    return ret_box       

def center_filtering_test(ct_heat,ht_box,ht_box_cls,kernel=3,K=70): # skip regressor part
    ct_heat = ct_heat.unsqueeze(0)    #(batch = 1,80,128,128)
    ct_heat = torch.sigmoid(ct_heat)   
    ct_heat = _nms(ct_heat, kernel=kernel) #(batch = 1,80,128,128)
    ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = _topk(ct_heat, K=K)   # all size with (batch = 1,K)
    ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = ct_scores[0], ct_inds[0], ct_clses[0], ct_ys[0], ct_xs[0]   # all size with (K,)
    
    n = 3
    ct_box = ht_box.new_full(ht_box[:,:4].shape,0)  #(number_of_pos_pred,4)
    ct_box[:,0] = ((n+1)*ht_box[:,0] + (n-1)*ht_box[:,2])/(2*n) #ctlx (number_of_pos_pred,)
    ct_box[:,1] = ((n+1)*ht_box[:,1] + (n-1)*ht_box[:,3])/(2*n) #ctly (number_of_pos_pred,)
    ct_box[:,2] = ((n-1)*ht_box[:,0] + (n+1)*ht_box[:,2])/(2*n) #cbrx (number_of_pos_pred,)
    ct_box[:,3] = ((n-1)*ht_box[:,1] + (n+1)*ht_box[:,3])/(2*n) #cbry (number_of_pos_pred,)
    ct_box_score = ht_box[:,4].clone()  #  (number_of_pos_pred,) 
    ct_box_cls = ht_box_cls #(number_of_pos_pred,)
    
    proposal_ind = ht_box.new_full(ht_box[:,4].shape,False,dtype = torch.uint8) #(number_of_pos_pred,)
    proposal_ct_score = ht_box.new_full(ht_box[:,4].shape,0) #(number_of_pos_pred,)
    
    for i in range(K):
        cty,ctx = ct_ys[i], ct_xs[i] # all size with (1,)
        ct_cls = ct_clses[i] # (1,)
        ct_score = ct_scores[i] # (1,)
        
        true = torch.as_tensor(True,device = ct_box.get_device())
        false = torch.as_tensor(False,device = ct_box.get_device())
        ct_tpk_filter = torch.where(ct_box_cls == ct_cls,true,false) & torch.where(ct_box[:,0]<=ctx,true,false) & torch.where(ct_box[:,1]<=cty,true,false) & torch.where(ct_box[:,2]>=ctx,true,false) & torch.where(ct_box[:,3]>=cty,true,false)#(number_of_pos_pred,)
        proposal_ind |= ct_tpk_filter #(number_of_pos_pred,)
        proposal_ct_score[ct_tpk_filter] = torch.max(proposal_ct_score[ct_tpk_filter],ct_score) #  (number_of_pos_pos_pred,)
    
    ret_box = ct_box[proposal_ind,:]  # (number_of_pos_pos_pred,4)
    ct_box_score[proposal_ind] = (ct_box_score[proposal_ind] * 2 + proposal_ct_score[proposal_ind])/3 # (number_of_pos_pos_pred,)
    neg_inds = ~proposal_ind  #(number_of_pos_pred,)
    ret_score = ct_box_score - neg_inds.float()  #  (number_of_pos_pred,)
    
    return ret_box, ret_score

def _neg_loss(preds, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)
        
    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        
    return loss

def _sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return x

def _ae_loss(tag0, tag1, mask):
    num  = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()

    tag_mean = (tag0 + tag1) / 2

    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    pull = tag0 + tag1

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num  = num.unsqueeze(2)
    num2 = (num - 1) * num
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push


def _regr_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _regr_l1_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    #regr    = regr[mask]
    #gt_regr = gt_regr[mask]
    
    #regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = F.l1_loss(regr * mask, gt_regr * mask, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """
    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious
