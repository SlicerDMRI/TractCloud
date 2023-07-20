from __future__ import print_function
import time
import torch.utils.data as data
import torch
import numpy as np
import pickle
import sys
import os
import whitematteranalysis as wma
from pytorch3d.transforms import RotateAxisAngle, Scale, Translate

sys.path.append('../')
import utils.tract_feat as tract_feat
from utils.funcs import obtain_TractClusterMapping, cluster2tract_label,\
    get_rot_axi, array2vtkPolyData, makepath
from utils.fiber_distance import MDF_distance_calculation, MDF_distance_calculation_endpoints   

class RealData_PatchData(data.Dataset):
    def __init__(self, feat, k, k_global, cal_equiv_dist=False, use_endpoints_dist=False, 
                 rough_num_fiber_each_iter=10000, k_ds_rate=0.1):
        self.feat = feat.astype(np.float32)   # [n_fiber, n_point, n_feat]
        self.k = k
        self.k_global = k_global
        self.cal_equiv_dist = cal_equiv_dist
        self.use_endpoints_dist = use_endpoints_dist
        self.rough_num_fiber_each_iter = rough_num_fiber_each_iter  # this will be a rough targeted number of fibers in each iteration.
        self.k_ds_rate=k_ds_rate
        
        num_fiber = self.feat.shape[0]
        num_point = self.feat.shape[1]
        num_feat_per_point = self.feat.shape[2]
        
        if self.k_global==0:
            self.global_feat = np.zeros((1, num_point, num_feat_per_point, 1), dtype=np.float32) # [1, n_point, n_feat, 1]
        else:
            random_idx = np.random.randint(0, num_fiber, self.k_global)
            self.global_feat = self.feat[random_idx,...]  # [k_global, n_point, n_feat]. This is the global (random) feature for all fibers in a test subject
            self.global_feat = self.global_feat.transpose(1,2,0)[None,:,:,:].astype(np.float32)  # [1, n_point, n_feat, k_global] 
        
        if self.k==0:
            self.local_feat = np.zeros((num_fiber, num_point, num_feat_per_point, 1), dtype=np.float32) # [n_fiber, n_point, n_feat, 1]
        else:
            self.local_feat = np.zeros((num_fiber, num_point, num_feat_per_point, self.k), dtype=np.float32)  # [n_fiber, k, n_point, n_feat]
            num_iter = num_fiber // self.rough_num_fiber_each_iter
            self.num_fiber_each_iter = (num_fiber // num_iter) + 1
            for i_iter in range(num_iter):
                cur_feat = self.feat[i_iter*self.num_fiber_each_iter:(i_iter+1)*self.num_fiber_each_iter,...]  # [n_fiber, n_point, n_feat]
                cur_feat = np.transpose(cur_feat,(0,2,1))  #  [n_fiber, n_point, n_feat]->[n_fiber,n_feat,n_point]
                cur_local_feat = cal_local_feat(cur_feat, self.k_ds_rate, self.k, self.use_endpoints_dist, self.cal_equiv_dist)      # [n_fiber*k, n_feat, n_point]
                cur_local_feat = cur_local_feat.reshape(cur_feat.shape[0], self.k, num_feat_per_point, num_point)  # [n_fiber, k, n_feat, n_point]
                cur_local_feat = np.transpose(cur_local_feat,(0,3,2,1))  # [n_fiber, k, n_feat, n_point]->[n_fiber, n_point, n_feat, k]
                self.local_feat[i_iter*self.num_fiber_each_iter:(i_iter+1)*self.num_fiber_each_iter,...] = cur_local_feat
            
            
    def __getitem__(self, index):
        point_set = self.feat[index]    # [n_point, n_feat]
        klocal_point_set = self.local_feat[index]   # [n_point, n_feat, k]

        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
            klocal_point_set = torch.from_numpy(klocal_point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))
            klocal_point_set = torch.from_numpy(klocal_point_set.astype(np.float32))
            print('Feature is not in float32 format')

        return point_set, klocal_point_set

    def __len__(self):
        return self.feat.shape[0]


class unrelatedHCP_PatchData(data.Dataset):
    def __init__(self, root, out_path, logger, split='train', num_fiber_per_brain=10000,num_point_per_fiber=15, 
                 use_tracts_training=False, k=0, k_global=0, rot_ang_lst=[0,0,0], scale_ratio_range=[0,0], trans_dis=0.0,
                 aug_axis_lst=['LR','AP', 'SI'], aug_times=10, cal_equiv_dist=False, k_ds_rate=0.1, recenter=False, include_org_data=False):        
        self.root = root
        self.out_path = out_path
        self.split = split
        self.logger = logger
        self.num_fiber = num_fiber_per_brain
        self.num_point = num_point_per_fiber
        self.use_tracts_training = use_tracts_training
        self.k = k
        self.k_global = k_global
        self.rot_ang_lst = rot_ang_lst
        self.scale_ratio_range = scale_ratio_range
        self.trans_dis = trans_dis
        self.aug_axis_lst = aug_axis_lst
        self.aug_times = aug_times
        self.k_ds_rate=k_ds_rate  
        self.recenter = recenter
        self.include_org_data = include_org_data
        
        # data save for debugging
        self.save_aug_data = True
        
        # algorithm tests
        self.cal_equiv_dist = cal_equiv_dist
        self.use_endpoints_dist = False
        self.logger.info('cal_equiv_dist: {}, use_endpoints_dist: {}'
                    .format(self.cal_equiv_dist, self.use_endpoints_dist))
                
        # load data
        with open(os.path.join(root, '{}.pickle'.format(split)), 'rb') as file:
            # Load the data from the file
            data_dict = pickle.load(file)
        self.features = data_dict['feat']
        self.labels = data_dict['label']
        self.label_names = data_dict['label_name']
        self.subject_ids = data_dict['subject_id']
        self.logger.info('Load {} data'.format(self.split))
        
        # calculate brain-level features  [n_subject, n_fiber, n_point, n_feat], labels [n_subject, n_fiber, n_point or 1]
        self.brain_features, self.brain_labels = self._cal_brain_feat()
        
        # calculate local global features/representations [n_subject*n_fiber, n_point, n_feat], [n_subject*n_fiber, n_point or 1], [n_subject*n_fiber, n_point, n_feat, k]
        # [n_subject*n_fiber, n_point, n_feat], [n_subject*n_fiber, n_point or 1], [n_subject*n_fiber, n_point, n_feat, k], [n_subject, n_point, n_feat, k_global], [n_subject*n_fiber, 1]
        self.org_feat, self.org_label, self.local_feat, self.global_feat, self.new_subidx = self._cal_info_feat()
        

    def __getitem__(self, index):
        point_set = self.org_feat[index]
        label = self.org_label[index]
        klocal_point_set = self.local_feat[index]
        new_subidx = self.new_subidx[index]

        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
            klocal_point_set = torch.from_numpy(klocal_point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))
            klocal_point_set = torch.from_numpy(klocal_point_set.astype(np.float32))
            print('Feature is not in float32 format')

        if label.dtype == 'int64':
            label = torch.from_numpy(label)
            new_subidx = torch.from_numpy(new_subidx)
        else:
            label = torch.from_numpy(label.astype(np.int64))
            new_subidx = torch.from_numpy(new_subidx.astype(np.int64))
            print('Label is not in int64 format')

        return point_set, label, klocal_point_set, new_subidx

    def __len__(self):
        return self.org_feat.shape[0]


    def _cal_brain_feat(self):
        """Process data for classification and segmentation. Get data in both brain and streamline level.
           Brain features are used for calculating the local-global representation.

        Args (self):
            features (array):[n_fiber, n_point, n_feat] original feature in fiber-level 
            subject_ids (array): [n_fiber] subject id for streamlines
            num_fiber_per_brain (int): number of fiber per brain
            num_point_per_fiber (int): number of point per fiber
            use_tracts_training (bool): whether to use tract label for training
        Returns:
            brain_feature (array):[n_subject, n_fiber, n_point, n_feat] feature in brain-level 
            brain_label (array): [n_subject, n_fiber, n_point or 1] label in brain-level
        """ 
        
        num_feat_per_point = self.features.shape[2]
        unique_subject_ids = np.unique(self.subject_ids)
        num_subject = len(unique_subject_ids)
        
        if self.aug_times > 0: # augmented data
            brain_features = np.zeros((num_subject*self.aug_times, self.num_fiber, self.num_point, num_feat_per_point),dtype=np.float32)
            brain_labels = np.zeros((num_subject*self.aug_times, self.num_fiber,1), dtype=np.int64)
            aug_matrices = np.zeros((num_subject, self.aug_times, 4, 4), dtype=np.float32)
        else:  # non-augmented data
            brain_features = np.zeros((num_subject, self.num_fiber, self.num_point, num_feat_per_point),dtype=np.float32)
            brain_labels = np.zeros((num_subject, self.num_fiber,1), dtype=np.int64)
                
                
        for i_subject, unique_id in enumerate(unique_subject_ids):  # for each subject
            cur_idxs = np.where(self.subject_ids == unique_id)[0]
            np.random.shuffle(cur_idxs)
            cur_select_idxs = cur_idxs[:self.num_fiber]
            cur_features = self.features[cur_select_idxs,:,:]  # [num_fiber_per_brain, num_point_per_fiber, num_feat_per_point]
            cur_labels = self.labels[cur_select_idxs, None]
            
            if self.aug_times > 0:
                # Augmentation the brain. Note that torch.from_numpy and .numpy() return data sharing the same memory location
                cur_features = torch.from_numpy(cur_features)  # numpy to tensor
                aug_features = np.zeros((self.aug_times, *cur_features.shape))  # (aug_times, num_fiber_per_brain, num_point_per_fiber, num_feat_per_point)  
                for i_aug in range(self.aug_times):
                    trot = None
                    cur_angles = []
                    # rotations
                    for i, rot_ang in enumerate(self.rot_ang_lst):
                        angle = ((torch.rand(1) - 0.5)*2*rot_ang).item()  # random angle between [-rot_ang, rot_ang] 
                        rot_axis_name = get_rot_axi(self.aug_axis_lst[i])
                        cur_trot = RotateAxisAngle(angle=angle, axis=rot_axis_name, degrees=True)  #  rotate around the axis by the angle
                        cur_angles.append(round(angle,1))
                        if trot is None:
                            trot = cur_trot
                        else:
                            trot = trot.compose(cur_trot)
                            
                    # scales
                    if self.scale_ratio_range[0] == 0 and self.scale_ratio_range[1] == 0:
                        scale_r = 1.0
                    else:
                        scale_r = torch.distributions.Uniform(1-self.scale_ratio_range[0], 1+self.scale_ratio_range[1]).sample().item()  # random scale between [1-scale_ratio_range[0], 1+scale_ratio_range[1]]
                    cur_trot = Scale(scale_r) 
                    trot = trot.compose(cur_trot)
                        
                    # translations
                    LR_trans = ((torch.rand(1) - 0.5)*2*self.trans_dis).item() # random translation between [-trans_dis, +trans_dis]
                    AP_trans = ((torch.rand(1) - 0.5)*2*self.trans_dis).item() # random translation between [-trans_dis, +trans_dis] 
                    SI_trans = ((torch.rand(1) - 0.5)*2*self.trans_dis).item() # random translation between [-trans_dis, +trans_dis]
                    cur_trot = Translate(LR_trans, AP_trans, SI_trans)
                    trot = trot.compose(cur_trot)
                        
                    aug_matrices[i_subject,i_aug,:,:] = np.array(trot.get_matrix())
                    aug_feat = trot.transform_points(cur_features).numpy()  # rotate and then convert tensor to numpy
                    
                    scale_r, LR_trans, AP_trans, SI_trans = round(scale_r,3),round(LR_trans,1),round(AP_trans,1),round(SI_trans,1)
                    if self.recenter:
                        aug_feat = center_tractography(self.root, aug_feat)
                        self.logger.info('Subject idx {} (unique ID {}, aug {}): rotation {}, scale {}, translation {} (centered). Aug axis order: {}'
                                        .format(i_subject, unique_id, i_aug, cur_angles, scale_r, [LR_trans, AP_trans, SI_trans], self.aug_axis_lst))
                    else:
                        self.logger.info('Subject idx {} (unique ID {}, aug {}): rotation {}, scale {}, translation {}. Aug axis order: {}'
                                        .format(i_subject, unique_id, i_aug, cur_angles, scale_r, [LR_trans, AP_trans, SI_trans], self.aug_axis_lst))
                    aug_features[i_aug,...] = aug_feat
                    # save augmented data
                    if self.save_aug_data and i_subject < 5: # only save the first 5 subjects
                        aug_data_save_path = os.path.join(self.out_path,'AugmentedData',self.split)
                        makepath(aug_data_save_path)
                        aug_feat_pd = array2vtkPolyData(aug_feat)
                        if self.recenter:
                            aug_feat_name = 'SubID{}Aug{}_RotR{}A{}S{}_Scale{}_TransR{}A{}S{}_Recenter'\
                                .format(i_subject, i_aug, cur_angles[0],cur_angles[1],cur_angles[2],
                                        scale_r, LR_trans, AP_trans, SI_trans)
                        else:
                            aug_feat_name = 'SubID{}Aug{}_RotR{}A{}S{}_Scale{}_TransR{}A{}S{}'\
                                .format(i_subject, i_aug, cur_angles[0],cur_angles[1],cur_angles[2],
                                        scale_r, LR_trans, AP_trans, SI_trans)                           
                        aug_feat_name = aug_feat_name.replace('.', '`') + '.vtk'
                        wma.io.write_polydata(aug_feat_pd, os.path.join(aug_data_save_path,aug_feat_name))  
                        print('Save augmented data to {}'.format(os.path.join(aug_data_save_path,aug_feat_name)))
    
                brain_features[i_subject*self.aug_times:(i_subject+1)*self.aug_times, :,:,:] = aug_features
            
            else:
                brain_features[i_subject,:,:,:] = cur_features
            
            if self.use_tracts_training:
                # map cluster label to tract label
                ordered_tract_cluster_mapping_dict = obtain_TractClusterMapping()  # {'tract name': ['cluster_xxx','cluster_xxx', ... 'cluster_xxx']} 
                cur_labels = cluster2tract_label(cur_labels, ordered_tract_cluster_mapping_dict, output_lst=False)
            if self.aug_times > 0:
                brain_labels[i_subject*self.aug_times:(i_subject+1)*self.aug_times,...] = cur_labels[None,...].repeat(self.aug_times, axis=0)  # [aug_times, num_fiber_per_brain, num_point_per_fiber or 1]
            else:
                brain_labels[i_subject,...] = cur_labels
        
        if self.aug_times > 0:      
            # save augmentation matrices
            np.save(os.path.join(self.out_path, '{}_aug_matrices.npy'.format(self.split)), aug_matrices)
            if self.include_org_data:
                assert self.num_fiber == 10000 # only support 10000 fibers for now, since each original brain has 10000 fibers
                org_features = self.features.reshape(num_subject, self.num_fiber, self.num_point, num_feat_per_point)
                org_labels = self.labels.reshape(num_subject, self.num_fiber, 1) 
                brain_features = np.concatenate((brain_features, org_features), axis=0)
                brain_labels = np.concatenate((brain_labels, org_labels), axis=0)
                self.logger.info('Include {} original data in the {} data.'.format(org_features.shape[0], self.split))
        
        return brain_features, brain_labels
        
        
    def _cal_info_feat(self):
        """
        Calculate local-global representations
        Args (self):
            n_subject=num_unique_subjects*self.aug_times
            brain_feature (array):[n_subject, n_fiber, n_point, n_feat] feature in brain-level 
            brain_label (array): [n_subject, n_fiber, n_point or 1] label in brain-level
            k (int, optional): How many k nearest neighbors are needed.
        Returns:
            fiber_feat (array): [n_subject*n_fiber, n_point, n_feat] feature in fiber-level
            fiber_label (array): [n_subject*n_fiber, n_point or 1] label in fiber-level
            local_feat (array): [n_subject*n_fiber, n_point, n_feat, k] k nearest neighbor streamline feature (local)
            global_feat (array): [n_subject, n_point, n_feat, 1] randomly selected streamline feature (global)
        """                   
        
        num_subjects = self.brain_features.shape[0]
        num_feat_per_point = self.brain_features.shape[-1]
        if self.k>0:
            local_feat = np.zeros((*self.brain_features.shape, self.k), dtype=np.float32) # [n_subject, n_fiber, n_point, n_feat, k]
        else: # will be discarded later in the training
            local_feat = np.zeros((*self.brain_features.shape, 1), dtype=np.float32) # [n_subject, n_fiber, n_point, n_feat, 1]
            local_feat = local_feat.reshape(-1, self.num_point, num_feat_per_point, 1)  # [n_subject*n_fiber, n_point, n_feat, 1]
        if self.k_global>0:
            global_feat = np.zeros((num_subjects, self.num_point, num_feat_per_point, self.k_global), dtype=np.float32) # [n_subject, n_point, n_feat, k_global]
        else:
            global_feat = np.zeros((num_subjects, self.num_point, num_feat_per_point, 1), dtype=np.float32) # [n_subject, n_point, n_feat, 1]
        # calculate new sub idx no where what the value of k and k_global are.
        new_subidx = np.zeros((num_subjects, self.num_fiber), dtype=np.int64)  # [n_subject, n_fiber]
        # iterate over augmented subjects
        for cur_idx in range(num_subjects):
            time_start = time.time()
            cur_feat = self.brain_features[cur_idx,...]  # [n_fiber,n_point,n_feat]
            cur_feat = np.transpose(cur_feat,(0,2,1))  #  [n_fiber, n_point, n_feat]->[n_fiber,n_feat,n_point]
            if self.k>0:
                # local feat
                cur_local_feat = cal_local_feat(cur_feat, self.k_ds_rate, self.k, self.use_endpoints_dist, self.cal_equiv_dist)      # [n_fiber*k, n_feat, n_point]
                cur_local_feat = cur_local_feat.reshape(self.num_fiber, self.k, num_feat_per_point, self.num_point)  # [n_fiber, k, n_feat, n_point]
                cur_local_feat = np.transpose(cur_local_feat,(0,3,2,1))  # [n_fiber, k, n_feat, n_point]->[n_fiber, n_point, n_feat, k]
            if self.k_global>0:
                # global feat
                random_idx = np.random.randint(0, cur_feat.shape[0], self.k_global)
                cur_global_feat = cur_feat[random_idx,...]  # [k_global, n_feat, n_point]. This is the random feature for all fibers in a test subject
                cur_global_feat = cur_global_feat.transpose(2,1,0)  # [n_point, n_feat, k_global]
            # new sub idx
            cur_subidx = np.ones((cur_feat.shape[0]), dtype=np.int64)*cur_idx   # [n_fiber,]
            time_end = time.time()
            if self.aug_times >0:
                self.logger.info('Subject {} Aug {} with {} fibers feature calculation time: {:.2f} s'
                                .format(cur_idx//self.aug_times, cur_idx%self.aug_times, self.num_fiber, time_end-time_start))
            else:
                self.logger.info('Subject {} (No Aug) with {} fibers feature calculation time: {:.2f} s'
                                .format(cur_idx, self.num_fiber, time_end-time_start))
            if self.k>0:
                local_feat[cur_idx,...] = cur_local_feat
            if self.k_global>0:
                global_feat[cur_idx,...] = cur_global_feat
            new_subidx[cur_idx,...] = cur_subidx  
                
        if self.k>0:
            local_feat = local_feat.reshape(-1, self.num_point, num_feat_per_point, self.k)  # [n_subject*n_fiber, n_point, n_feat, k]
        new_subidx = new_subidx.reshape(-1, 1)  # [n_subject*n_fiber, 1]
        
        # original features and labels
        fiber_feat = self.brain_features.reshape(-1, self.num_point, num_feat_per_point)  # [n_subject*n_fiber, n_point, n_feat]
        fiber_label = self.brain_labels.reshape(-1, 1)  # [n_subject*n_fiber, 1]
        
        return fiber_feat, fiber_label, local_feat, global_feat, new_subidx
    


def cal_local_feat(cur_feat, k_ds_rate, k, use_endpoints_dist, cal_equiv_dist):
    """ Calculate the local feature for all streamlines in cur_feat
    Args:
        cur_feat: [n_fiber, n_feat, n_point]
        k_ds_rate: the rate of downsample the fibers to calculate the distance matrix. 1 means no downsample
        k: the number of nearest neighbor streamlines (local)
        use_endpoints_dist (bool): whether to use the distance between endpoints to calculate the distance matrix
        cal_equiv_dist (bool): whether to calculate the equivalent distance matrix

    Returns:
        cur_local_feat: [n_fiber*k, n_feat, n_point], features of the k nearest neighbor streamlines
    """
    # local feat
    # [n_fiber, k], [n_fiber, k], [n_fiber,n_feat,n_point], [n_fiber,n_feat,n_point]
    near_idx, near_flip_mask, ds_cur_feat, ds_cur_feat_equiv = dist_mat_knn(
                        torch.from_numpy(cur_feat), k_ds_rate, k, use_endpoints_dist, cal_equiv_dist)
    cur_local_feat_org = ds_cur_feat[near_idx.reshape(-1),...]  # [n_fiber*k, n_feat, n_point]
    cur_local_feat_equiv = ds_cur_feat_equiv[near_idx.reshape(-1),...]  # [n_fiber*k, n_feat, n_point]
    near_flip_mask = near_flip_mask.reshape(-1)[:,None,None]  # [n_fiber*k, 1, 1]
    near_nonflip_mask = 1-near_flip_mask
    cur_local_feat = cur_local_feat_org*near_nonflip_mask + cur_local_feat_equiv*near_flip_mask  # [n_fiber*k, n_feat, n_point]

    return cur_local_feat 


def dist_mat_knn(brain_feat, k_ds_rate, k, use_endpoints_dist, cal_equiv_dist):
    """ 
        calculate the distance between streamlines (fibers) and then find the neighbor streamlines (fibers)
        input (self) 
            brain_feat: [n_fiber, n_feat, n_point]
            k_ds_rate (float): the rate of downsample the fibers to calculate the distance matrix
            k (int): the number of nearest neighbors
            use_endpoints_dist (bool): whether to use endpoints distance to calculate the nearest neighbor streamlines (fibers)
            cal_equiv_dist (bool): whether to calculate the equivalent distance
        output 
            idx (the nearest idx): [n_fiber, k]
            flip_mask (the flip mask): [n_fiber, k]. whether the nearest fiber feature should use flipped fiber (reverse order)
            ds_brain_feat (the downsampled brain_feat): [n_fiber, n_feat, n_point]
            ds_brain_feat_equiv (the downsampled equivalent (reverse order) brain_feat): [n_fiber, n_feat, n_point]
    """
        
    # calculate the distance matrix. Take the minus since we wanna find the smallest distance using topk.
    if 0 < k_ds_rate < 1:
        num_ds_feat =  int(brain_feat.shape[0]*k_ds_rate)
        ds_indices = np.random.choice(brain_feat.shape[0], size=num_ds_feat, replace=False)
        downsample_feat = brain_feat[ds_indices,:,:]  # [n_ds_fiber, n_point, n_feat]
    else:
        downsample_feat = brain_feat
    if use_endpoints_dist:
        dist_mat, flip_mask, ds_brain_feat, ds_brain_feat_equiv = MDF_distance_calculation_endpoints(brain_feat, downsample_feat, cal_equiv=cal_equiv_dist)  # (n_fiber, n_ds_fiber) for dist_mat
    else:
        dist_mat, flip_mask, ds_brain_feat, ds_brain_feat_equiv = MDF_distance_calculation(brain_feat, downsample_feat, cal_equiv=cal_equiv_dist)  # (n_fiber, n_ds_fiber)
    
    topk_idx = dist_mat.topk(k=k, largest=False, dim=-1)[1]   # (N_fiber, k). largest is False then the k smallest elements are returned.
    near_idx = topk_idx[:,:]   # (N_fiber, k)
    # print(dist_mat[0,near_idx[0,:]])
    near_flip_mask = torch.gather(flip_mask, dim=1, index=near_idx) # (N_fiber, k). The flip mask of the info fibers (neighbor).

    return near_idx.numpy(), near_flip_mask.numpy(), ds_brain_feat.numpy(), ds_brain_feat_equiv.numpy()

        
def center_tractography(input_path, feat_RAS, out_path=None, logger=None, tractography_name=None,save_data=False):
    """Recenter the tractography to atlas center
        feat_RAS: [n_fiber, n_point, n_feat]"""
    HCP_center = np.load(os.path.join(input_path, 'HCP_mass_center.npy'))  # (15(n_point),3(n_feat)) from 100 unrelated HCP subjects (atlas). The calculation function is in func_intra.py
    test_subject_center = np.mean(feat_RAS, axis=0)
    displacement = HCP_center - test_subject_center
    c_feat_RAS = feat_RAS + displacement  # recenter the tractography to HCP atlas center
    if save_data:
        recenter_path = os.path.join(out_path, 'recentered_tractography')
        makepath(recenter_path)
        feat_RAS_pd = array2vtkPolyData(c_feat_RAS)
        wma.io.write_polydata(feat_RAS_pd, os.path.join(recenter_path, 'recentered_{}'.format(tractography_name)))
        logger.info('Saved recentered tractography to {}'.format(recenter_path))
    return c_feat_RAS
