import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .py_utils import TopPool, BottomPool, LeftPool, RightPool
from .py_utils import hg52, AELoss, _neg_loss, convolution, residual, make_kp_layer
from .py_utils.HG52 import AELoss52

class pool(nn.Module):
    def __init__(self, dim, pool1, pool2):
        super(pool, self).__init__()
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()

        self.look_conv1 = convolution(3, dim, 128)
        self.look_conv2 = convolution(3, dim, 128)
        self.P1_look_conv = nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=False)
        self.P2_look_conv = nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=False)

    def forward(self, x):
        # pool 1
        look_conv1   = self.look_conv1(x)
        p1_conv1     = self.p1_conv1(x)
        look_right   = self.pool2(look_conv1)
        P1_look_conv = self.P1_look_conv(p1_conv1+look_right)
        pool1        = self.pool1(P1_look_conv)

        # pool 2
        look_conv2   = self.look_conv2(x)
        p2_conv1     = self.p2_conv1(x)
        look_down    = self.pool1(look_conv2)
        P2_look_conv = self.P2_look_conv(p2_conv1+look_down)
        pool2        = self.pool2(P2_look_conv)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2

class regionconv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(regionconv, self).__init__()
        self.conv1 = convolution(3, in_dim, in_dim)
        self.bn1   = nn.BatchNorm2d(in_dim)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = convolution(3, in_dim, out_dim)
        self.bn2   = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)
        
        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)
        return bn2


class tl_pool(pool):
    def __init__(self, dim):
        super(tl_pool, self).__init__(dim, TopPool, LeftPool)

class br_pool(pool):
    def __init__(self, dim):
        super(br_pool, self).__init__(dim, BottomPool, RightPool)

class region_conv(regionconv):
    def __init__(self, in_dim, out_dim):
        super(region_conv, self).__init__(in_dim, out_dim)

def make_tl_layer(dim):
    return tl_pool(dim)

def make_br_layer(dim):
    return br_pool(dim)

def make_region_layer(in_dim, out_dim):
    return region_conv(in_dim, out_dim)

def make_pool_layer(dim):
    return nn.Sequential()

def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers  = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1) for _ in range(mod - 1)]
    return nn.Sequential(*layers)

class model(hg52):
    def __init__(self, db):
        n       = 5
        dims    = [256, 256, 384, 384, 384, 512]
        modules = [2, 2, 2, 2, 2, 4]
        #out_dim = 80
        out_dim = 1

        super(model, self).__init__(
            db, n, 1, dims, modules, out_dim,
            make_tl_layer=make_tl_layer,
            make_br_layer=make_br_layer,
            make_region_layer=make_region_layer,
            make_grouping_layer=make_region_layer,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            kp_layer=residual, cnv_dim=256
        )

loss = AELoss52(pull_weight=1e-1, push_weight=1e-1, focal_loss=_neg_loss)


