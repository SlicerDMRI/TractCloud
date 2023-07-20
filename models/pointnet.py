"""
Borrow from https://github.com/fxia22/pointnet.pytorch

Modified by 
@Author: Tengfei Xue
@Contact: txue4133@uni.sydney.edu.au
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, k=0, k_global=0, global_feat = True, feature_transform = False, first_feature_transform=False):
        super(PointNetfeat, self).__init__()
        if k+k_global == 0:  
            self.conv1 = torch.nn.Conv1d(3, 64, 1)
        else:
            self.info_conv = torch.nn.Conv2d(3*2, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        if k+k_global == 0:
            self.bn1 = nn.BatchNorm1d(64)
        else:
            self.info_bn = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        
        self.first_feature_transform = first_feature_transform
        self.feature_transform = feature_transform
        self.k = k
        self.k_global = k_global
        if self.first_feature_transform:
            self.stn = STN3d()
        if self.feature_transform:
            self.fstn = STNkd(k=64)
            

    def forward(self, x, info_point_set):
        n_pts = x.size()[2]
        
        if self.first_feature_transform:
            trans = self.stn(x) 
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        else:
            trans = None
        
        if self.k + self.k_global == 0:
            #* process each streamline individually, no local+global info
            x = F.relu(self.bn1(self.conv1(x)))
        else:
            #* input local+global info for each streamline
            x = x[:,:,:,None].repeat(1,1,1,self.k+self.k_global)    # (num_fiber, 3, num_points) -> (num_fiber, 3, num_points, fiber_k)
            x = torch.cat((info_point_set-x, x),dim=1)   #  (num_fiber, 3*2, num_points, fiber_k)
            x = F.relu(self.info_bn(self.info_conv(x)))      # (num_fiber, 3*2, num_points, fiber_k) -> (num_fiber, 64, num_points, fiber_k)
            x = x.max(dim=-1, keepdim=False)[0]    # (num_fiber, 64,num_points, fiber_k) -> (num_fiber,64, num_points)
        
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=0, k_global=0, num_classes=2, feature_transform=False, first_feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.first_feature_transform = first_feature_transform
        self.k = k
        self.k_global = k_global
        self.feat = PointNetfeat(k=k, k_global=k_global, global_feat=True, feature_transform=feature_transform, first_feature_transform=first_feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        # self.dropout = nn.Dropout(p=dropout)  # todo: add dropout param that can be tuned outside
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x, info_point_set):
        """x (num_fiber, 3, num_points)"""
        x, trans, trans_feat = self.feat(x, info_point_set)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat