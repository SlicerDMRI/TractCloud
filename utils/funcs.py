import os
import pickle
import vtk
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
import numpy as np
import pandas as pd
import random
import torch
from collections import defaultdict
import whitematteranalysis as wma


def round_decimal(value, decimal=4):
    """Round to 2 decimal
       0.9652132 to 0.9652"""
    decimal_zeros = ''
    for _ in range(decimal):
        decimal_zeros = '0' + decimal_zeros

    new_value = float(Decimal(str(value)).quantize(Decimal('0.{}'.format(decimal_zeros)), rounding=ROUND_HALF_EVEN))

    return new_value


def makepath(dir):
    try:
        os.makedirs(dir)
    except OSError:
        pass


def unify_path(path):
    """Remove '/' at the end, if it exists"""
    if path[-1] == '/':
        path = path[:-1]
    else:
        path = path

    return path


def fix_seed(manualSeed):
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)   # seed for cpu
    torch.cuda.manual_seed(manualSeed)  # seed for gpu
    torch.cuda.manual_seed_all(manualSeed)  # seed for all gpu
    

def obtain_TractFullName():    
    tract_name_full_dict = {
        'AF': 'arcuate fasciculus',  # association
        'CB': 'cingulum bundle',
        'EC': 'external capsule',
        'EmC': 'extreme capsule',
        'ILF': 'inferior longitudinal fasciculus',
        'IOFF': 'inferior occipito-frontal fasciculus',
        'MdLF': 'middle longitudinal fasciculus',
        'SLF-I': 'superior longitudinal fasciculus I',
        'SLF-II': 'superior longitudinal fasciculus II',
        'SLF-III': 'superior longitudinal fasciculus III',
        'UF': 'uncinate fasciculus',
        'CST':'corticospinal tract',  # projection
        'CR-F':'corona-radiata-frontal (excluding the CST)',
        'CR-P':'corona-radiata-parietal (excluding the CST)',
        'SF':'striato-frontal',
        'SO':'striato-occipital',
        'SP':'striato-parietal',
        'TF':'thalamo-frontal',
        'TO':'thalamo-occipital',
        'TT': 'thalamo-temporal',
        'TP': 'thalamo-parietal',
        'PLIC':'posterior limb of internal capsule',
        'CC1': 'corpus callosum 1',    # Commissural
        'CC2': 'corpus callosum 2',    
        'CC3': 'corpus callosum 3',    
        'CC4': 'corpus callosum 4',    
        'CC5': 'corpus callosum 5',    
        'CC6': 'corpus callosum 6',    
        'CC7': 'corpus callosum 7',
        'CPC': 'cortico-ponto-cerebellar',     # cerebellar
        'ICP': 'inferior cerebellar peduncle', 
        'Intra-CBLM-I-P':'intracerebellar input and Purkinje tract',
        'Intra-CBLM-PaT':'intracerebellar parallel tract',
        'MCP':'middle cerebellar peduncle',
        'Sup-F':'superficial-frontal',  # superficial
        'Sup-FP':'superficial-frontal-parietal',
        'Sup-O':'superficial-occipital',
        'Sup-OT':'superficial-occipital-temporal',
        'Sup-P':'superficial-parietal',
        'Sup-PO':'superficial-parietal-occipital',
        'Sup-PT':'superficial-parietal-temporal',
        'Sup-T':'superficial-temporal'
        }
    
    return tract_name_full_dict


def obtain_TractClusterMapping():
    """The obtained dictionary is {'tract name': ['cluster_xxx','cluster_xxx', ... 'cluster_xxx']}. Cluster index starts from 1."""
    ordered_tract_names = list(obtain_TractFullName().keys())
    ordered_tract_cluster_mapping_dict = {}   # keep the order we like association, projection, commissural, cerebellar, superficial 
    
    cluster_annotation_pd = pd.read_excel('../datasets/FiberClusterAnnotation_Updated20230110.xlsx')
    cluster_tract_mapping = {clu:tra for clu, tra in zip(cluster_annotation_pd['Cluster Index'], cluster_annotation_pd['Final'])}

    # only include clusters into the anatomical tracts
    tract_cluster_mapping_dict = defaultdict(list)  # {'tract_x': ['cluster_xxx','cluster_xxx']}
    for key, value in cluster_tract_mapping.items():
        tract_cluster_mapping_dict[value].append(key)   
    
    # sort the tract order in the dictionary and add clusters belong to other tracts
    all_clusters_lst = list(cluster_annotation_pd['Cluster Index'])
    clusters_belong_anatomical_tracts_lst = []
    for tract_name in ordered_tract_names:
        ordered_tract_cluster_mapping_dict[tract_name] = tract_cluster_mapping_dict[tract_name]
        clusters_belong_anatomical_tracts_lst.extend(tract_cluster_mapping_dict[tract_name])
    ordered_tract_cluster_mapping_dict['Other'] = sorted(list(set(all_clusters_lst)-set(clusters_belong_anatomical_tracts_lst))) 

    
    return ordered_tract_cluster_mapping_dict


def cluster2tract_label(lst, ordered_tract_cluster_mapping_dict, output_lst=True):
    """Convert cluster labels to tract labels"""
    if type(lst) != np.ndarray:
        array = np.array(lst)
    else:
        array = lst
        
    org_array = np.copy(array)   # The original label/pred lst
    
    num_tracts = len(ordered_tract_cluster_mapping_dict.keys())  # anatomically plausible tracts + 1 other tract
    
    # convert cluster idx to tract idx. Iterate tracts then clusters
    for idx_tract, tract_name in enumerate(ordered_tract_cluster_mapping_dict.keys()):
        cur_clusters_names = ordered_tract_cluster_mapping_dict[tract_name]
        for cluster_name in cur_clusters_names:
            cluster_idx = int(cluster_name.split('_')[1][2:])-1   # '00031'->30,'00322'->321.  Don't use strip('0'), it will strip 0 in '80' as well.
            array[org_array==cluster_idx] = idx_tract  # plausible tract, starts from 0
            array[org_array==cluster_idx+800] = num_tracts-1  # implausible tract (outlier)
    
    if output_lst:   # when array is 1D
        new_lst = list(array)
    else:           # We could get array with 2D/3D/nD shape, e.g., (num_fiber_per_brian, num_points_per_fiber)
        new_lst = array
    
    return new_lst       


def get_rot_axi(axis_name):
    if axis_name == 'LR':
        rot_axi = 'X'
    elif axis_name == 'AP':
        rot_axi = 'Y'
    elif axis_name == 'SI':
        rot_axi = 'Z'     
    return rot_axi


def array2vtkPolyData(array):
    """convert array to vtkPolyData
        array: (num_fibers, num_points, num_feat(3))
       from WMA https://github.com/SlicerDMRI/whitematteranalysis/blob/33a7b13f5b452a2c453c91619268c6539ae5a574/whitematteranalysis/fibers.py#L435"""
    num_streamlines = array.shape[0]
    num_points_per_s = array.shape[1]
    
    outpd = vtk.vtkPolyData()
    outpoints = vtk.vtkPoints()
    outlines = vtk.vtkCellArray()
    
    outlines.InitTraversal()

    for lidx in range(0, num_streamlines):
        cellptids = vtk.vtkIdList()
        
        for pidx in range(0, num_points_per_s):

            idx = outpoints.InsertNextPoint(array[lidx, pidx, :])

            cellptids.InsertNextId(idx)
        
        outlines.InsertNextCell(cellptids)
        
    # put data into output polydata
    outpd.SetLines(outlines)
    outpd.SetPoints(outpoints)
    
    return outpd      


def save_info_feat(org_feat, info_feat, new_subidx, start, end, aug_times, k, k_global, k_ds_rate, org_knn_feat_save_folder):
    for fiber_idx in range(start,end+1):
        org_feat = np.array(org_feat)
        info_feat = np.array(info_feat)
        new_subidx = np.array(new_subidx)
        sample_org_feat = org_feat[fiber_idx,None,:,:].transpose((0,2,1)) # [1, n_point, n_feat]
        sample_knn_feat = info_feat[fiber_idx,...].transpose((2,1,0)) # [k+k_global, n_point, n_feat]
        if aug_times >0:
            sub_id = new_subidx[fiber_idx] // aug_times
            aug_id =  new_subidx[fiber_idx] % aug_times
        else:
            sub_id = new_subidx[fiber_idx]
            aug_id = 0
        sample_org_pd = array2vtkPolyData(sample_org_feat)
        sample_org_save_path = os.path.join(org_knn_feat_save_folder,'SubID{}Aug{}FiberID{}_k{}_ds{}_ksparse{}_org_pd.vtk'
                                            .format(sub_id, aug_id, fiber_idx, k, str(k_ds_rate).replace('.','`'), k_global))
        wma.io.write_polydata(sample_org_pd, sample_org_save_path)
        print('Save org feature to {}'.format(sample_org_save_path))
        sample_knn_pd = array2vtkPolyData(sample_knn_feat)
        sample_knn_save_path = os.path.join(org_knn_feat_save_folder,'SubID{}Aug{}FiberID{}_k{}_ds{}_ksparse{}_knn_pd.vtk'
                                            .format(sub_id, aug_id, fiber_idx, k, str(k_ds_rate).replace('.','`'), k_global))
        wma.io.write_polydata(sample_knn_pd, sample_knn_save_path)
        print('Save knn feature to {}'.format(sample_knn_save_path))
        
        
def tractography_parcellation(args, pd_tractography, predicted_lst, label_names):
    """Generate the tractography parcellation results with the predicted list"""
    print('===================================')
    print('Output fiber clusters.')
    output_cluster_folder = os.path.join(args.out_path, 'predictions')
    makepath(output_cluster_folder)
    # Tractography Parcellation
    cluster_prediction_mask = np.array(predicted_lst)
    number_of_clusters = np.max(cluster_prediction_mask) + 1
    pd_t_list = wma.cluster.mask_all_clusters(pd_tractography, cluster_prediction_mask, number_of_clusters,
                                              preserve_point_data=False, preserve_cell_data=False, verbose=False)
    for t_idx in range(len(pd_t_list)):
        if label_names[t_idx] == 'Other':  # not save "Other" cluster/tract
            continue
        pd_t = pd_t_list[t_idx]
        fname_t = os.path.join(output_cluster_folder, label_names[t_idx] + '.vtp')
        print('output', fname_t)
        wma.io.write_polydata(pd_t, fname_t)

    print('Done! Clusters are in:', output_cluster_folder)
    
    
def str2num(rot_ang_lst_str):
    new_rot_ang_lst = list(map(float, rot_ang_lst_str.split("_")))
    return new_rot_ang_lst