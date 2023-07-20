"""Ref: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
https://discuss.pytorch.org/t/subtraction-along-axis/94795
"""

import torch
import numpy as np
import whitematteranalysis as wma
import time


def fiber_distance_cal_Efficient(set1, set2=None, num_points=15):
    '''
    Calculate distances between points using quadratic expansion. The most efficient way to calculate the distance between two point sets.
    Input: set1 is a Nxn_px3 matrix
           set2 is an optional Mxn_px3 matrix
    Output: dist is a NxM matrix where dist[i,j] is the square norm between set1[i,:,:] and set2[j,:,:]
            if set2 is not given then use 'set2=set1'.
    i.e. dist[i,j] = ||set1[i,:,:]-set2[j,:,:]||^2
    '''
    
    set1 = set1.reshape(set1.shape[0], -1)   # set1 [N, 3*n_p]
    set2 = set2.reshape(set2.shape[0], -1) if set2 is not None else set1  # set2 [M, 3*n_p]
    set1_squ = (set1 ** 2).sum(1).view(-1, 1)  # set1_squ [N, 1]
    set2_t = torch.transpose(set2, 0, 1)   # set2_t [3*n_p, M]
    set2_squ = (set2 ** 2).sum(1).view(1, -1)   # set2_squ [1, M]
    dist = set1_squ + set2_squ - 2.0 * torch.mm(set1, set2_t)  # dist [N, M]
    # Ensure diagonal is zero if set1=set2
    if set2 is None:
       dist = dist - torch.diag(dist.diag())  
    dist = torch.sqrt(torch.clamp(dist, 0.0, np.inf))
    
    mean_dist = torch.div(dist, num_points)
    
    return mean_dist



def MDF_distance_calculation(point_set,ds_point_set=None,cal_equiv=False):
    """minimum average direct-flip (MDF) distance (QuickBundles, Deep Fiber clustering)
        Calculate x,y,z distance separately and then add them together
        input:
            point_set: [num_fibers, 3, num_points]
            ds_point_set: [num_downsample_fibers, 3, num_points]
            cal_equiv: whether to calculate the flipped distance
        output:
            final_MDF_dist: [num_fibers, num_downsample_fibers]
            flip_mask: [num_fibers, num_downsample_fibers] 1: flip, 0: not flip"""
            
    # dowsample point set
    if ds_point_set is None:
        ds_point_set = point_set
    # calculate distance and flipped distance
    ds_point_set_equiv = torch.flip(ds_point_set, dims=[-1])    
    # Most efficient implementation
    time_start = time.time()
    new_mean_dist = fiber_distance_cal_Efficient(point_set, ds_point_set)
    if cal_equiv:
        new_mean_dist_equiv = fiber_distance_cal_Efficient(point_set, ds_point_set_equiv)
        new_MDF_dist = torch.minimum(new_mean_dist, new_mean_dist_equiv)
        new_flip_mask = torch.where(new_mean_dist_equiv < new_mean_dist, 1, 0)  # if equivalent distance is smaller than original distance, flip the fiber
    else:
        new_MDF_dist = new_mean_dist
        new_flip_mask = torch.zeros_like(new_MDF_dist)
    new_final_MDF_dist = new_MDF_dist    
    time_end = time.time()
    # print('New distance cal time cost', time_end - time_start, 's')
    return new_final_MDF_dist, new_flip_mask, ds_point_set, ds_point_set_equiv


def MDF_distance_calculation_endpoints(point_set, ds_point_set=None, cal_equiv=True):
    """minimum average direct-flip (MDF) distance (QuickBundles, Deep Fiber clustering)
        Only use two endpoints for distance calculation"""
        
    # dowsample point set
    if ds_point_set is None:
        ds_point_set = point_set
    # calculate distance and flipped distance
    ds_point_set_equiv = torch.flip(ds_point_set, dims=[-1])
    # Most efficient implementation
    time_start = time.time()
    new_mean_dist = fiber_distance_cal_Efficient(point_set[:, :, [0, -1]], ds_point_set[:, :, [0, -1]])
    if cal_equiv:
        new_mean_dist_equiv = fiber_distance_cal_Efficient(point_set[:, :, [0, -1]], ds_point_set_equiv[:, :, [0, -1]],)
        new_MDF_dist = torch.minimum(new_mean_dist, new_mean_dist_equiv)
        new_flip_mask = torch.where(new_mean_dist_equiv < new_mean_dist, 1, 0)  # if equivalent distance is smaller than original distance, flip the fiber
    else:
        new_MDF_dist = new_mean_dist
        new_flip_mask = torch.zeros_like(new_MDF_dist)
    new_final_MDF_dist = new_MDF_dist    
    time_end = time.time()
    print('New distance cal time cost', time_end - time_start, 's')
    return new_final_MDF_dist, new_flip_mask, ds_point_set, ds_point_set_equiv