"""
Created on Thu Apr 25 15:12:27 2021

@author: yang an
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, InstanceNorm2d

class cheby_conv(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''

    def __init__(self, c_in, c_out, K, Kt):
        super(cheby_conv, self).__init__()
        c_in_new = (K) * c_in
        self.conv1 = Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.K = K

    def forward(self, x, adj):
        nSample, feat_in, nNode, length = x.shape
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        # print(Lap)
        Lap = Lap.transpose(-1, -2)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out

class ST_BLOCK(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super(ST_BLOCK, self).__init__()
        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0, 1),
                            stride=(1, 1), bias=True)
        self.gcn = cheby_conv(c_out // 2, c_out, K, 1)
        self.conv2 = Conv2d(c_out, c_out * 2, kernel_size=(1, Kt), padding=(0, 1),
                            stride=(1, 1), bias=True)
        self.c_out = c_out
        self.conv_1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                             stride=(1, 1), bias=True)
        # self.conv_2=Conv2d(c_out//2, c_out, kernel_size=(1, 1),
        #                stride=(1,1), bias=True)

    def forward(self, x, supports):
        x_input1 = self.conv_1(x)
        x1 = self.conv1(x)
        filter1, gate1 = torch.split(x1, [self.c_out // 2, self.c_out // 2], 1)
        x1 = (filter1) * torch.sigmoid(gate1)
        x2 = self.gcn(x1, supports)
        x2 = torch.relu(x2)
        # x_input2=self.conv_2(x2)
        x3 = self.conv2(x2)
        filter2, gate2 = torch.split(x3, [self.c_out, self.c_out], 1)
        x = (filter2 + x_input1) * torch.sigmoid(gate2)
        return x

class Gated_STGCN(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, week, day, recent, K, Kt):
        super(Gated_STGCN, self).__init__()
        tem_size = week + day + recent
        self.block1 = ST_BLOCK(c_in, c_out, num_nodes, tem_size, K, Kt)
        self.block2 = ST_BLOCK(c_out, c_out, num_nodes, tem_size, K, Kt)
        self.block3 = ST_BLOCK(c_out, c_out, num_nodes, tem_size, K, Kt)

        self.bn = BatchNorm2d(c_in, affine=False)
        self.conv1 = Conv2d(c_out, 12, kernel_size=(1, recent), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.c_out = c_out

    def forward(self, x_w, x_d, x_r, supports):
        x = self.bn(x_r)
        shape = x.shape

        x = self.block1(x, supports)
        x = self.block2(x, supports)
        x = self.block3(x, supports)
        x = self.conv1(x).squeeze().permute(0, 2, 1).contiguous()  # b,n,l
        return x, supports, supports

    # 模型8

