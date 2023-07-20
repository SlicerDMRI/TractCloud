import argparse
import os
import sys
sys.path.append('../')
import time
import h5py
import numpy as np

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from datasets.dataset import unrelatedHCP_PatchData
from models.pointnet import PointNetCls
from models.dgcnn import tract_DGCNN_cls
from utils.logger import create_logger
from utils.metrics_plots import classify_report, process_curves, calculate_acc_prec_recall_f1, best_swap, save_best_weights
from utils.funcs import round_decimal, unify_path, makepath, fix_seed, obtain_TractClusterMapping, cluster2tract_label, save_info_feat, str2num
from utils.cli import create_parser, save_args, adaptive_args

def load_datasets(eval_split, args, test=False, logger=None):
    """load train and validation data"""
    # load feature and label data
    if not test:
        train_dataset = unrelatedHCP_PatchData(
                root=args.input_path,
                out_path=args.out_path,
                logger=logger,
                split='train',
                num_fiber_per_brain=args.num_fiber_per_brain,
                num_point_per_fiber=args.num_point_per_fiber,
                use_tracts_training=args.use_tracts_training,
                k=args.k,
                k_global=args.k_global,
                rot_ang_lst=args.rot_ang_lst,
                scale_ratio_range=args.scale_ratio_range,
                trans_dis=args.trans_dis,
                aug_times=args.aug_times,
                cal_equiv_dist=args.cal_equiv_dist,
                k_ds_rate=args.k_ds_rate,
                recenter=args.recenter,
                include_org_data=args.include_org_data)
    else:
        train_dataset = None
        
    eval_dataset = unrelatedHCP_PatchData(
        root=args.input_path,
        out_path=args.out_path,
        logger=logger,
        split=eval_split,
        num_fiber_per_brain=args.num_fiber_per_brain,
        num_point_per_fiber=args.num_point_per_fiber,
        use_tracts_training=args.use_tracts_training,
        k=args.k,
        k_global=args.k_global,
        rot_ang_lst=args.rot_ang_lst,
        scale_ratio_range=args.scale_ratio_range,
        trans_dis=args.trans_dis,
        aug_times=args.aug_times,
        cal_equiv_dist=args.cal_equiv_dist,
        k_ds_rate=args.k_ds_rate,
        recenter=args.recenter,
        include_org_data=args.include_org_data)

    return train_dataset, eval_dataset


def load_batch_data():
    """load train and val batch data"""
    eval_state='val'
    train_dataset, val_dataset = load_datasets(eval_split=eval_state, args=args, test=False, logger=logger)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size,
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size,
        shuffle=True)

    train_data_size = len(train_dataset)
    val_data_size = len(val_dataset)
    try:
        train_feat_shape = train_dataset.fiber_feat.shape
        val_feat_shape = val_dataset.fiber_feat.shape
    except:
        train_feat_shape = train_dataset.org_feat.shape
        val_feat_shape = val_dataset.org_feat.shape
    logger.info('The training data feature size is:{}'.format(train_feat_shape))
    logger.info('The validation data feature size is:{}'.format(val_feat_shape))
    
    # load label names
    if args.use_tracts_training:
        label_names =  list(ordered_tract_cluster_mapping_dict.keys()) 
    else:
        assert train_dataset.label_names == val_dataset.label_names
        label_names = train_dataset.label_names
    label_names_h5 = h5py.File(os.path.join(args.out_path, 'label_names.h5'), 'w')
    label_names_h5['y_names'] = label_names
    logger.info('The label names are: {}'.format(str(label_names)))
    num_classes = len(np.unique(label_names))
    logger.info('The number of classes is:{}'.format(num_classes))

    # global feature
    train_global_feat = train_dataset.global_feat
    val_global_feat = val_dataset.global_feat
    
    return train_loader, val_loader, label_names, num_classes, train_data_size, val_data_size, eval_state, train_global_feat, val_global_feat


def load_model(args, num_classes, device, test=False):
    # model setting 
    if args.model_name == 'dgcnn':
        DL_model = tract_DGCNN_cls(num_classes,args,device)
    elif args.model_name == 'pointnet':
        DL_model = PointNetCls(k=args.k, k_global=args.k_global, num_classes=num_classes, feature_transform=False, first_feature_transform=False)
    else:
        raise ValueError('Please input valid model name dgcnn | pointnet')
        
    # load weights when testing
    if test:
        weight = torch.load(args.weight_path)  
        DL_model.load_state_dict(weight)
            
    DL_model.to(device)

    return DL_model


def load_settings(DL_model):
    # optimizers
    if args.opt == 'Adam':
        optimizer = optim.Adam(DL_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.opt == 'SGD':
        optimizer = optim.SGD(DL_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError('Please input valid optimizers Adam | SGD')
    # schedulers
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.decay_factor)
    elif args.scheduler == 'wucd':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult)
    else:
        raise ValueError('Please input valid schedulers step | wucd')
    
    return optimizer, scheduler


def train_val_test_forward(idx_data, data, net, state, total_loss, labels_lst, predicted_lst, args, device, num_classes, epoch=-1, num_batch=-1,
                           train_global_feat=None, eval_global_feat=None):
    if state == 'test_realdata':
        points, klocal_feat_set = data
    else:
        # points [B, N_point, 3], label [B,1](cls) or [B,N_point](seg), [B, n_point, 3, k], [B]
        points, label, klocal_feat_set, new_subidx = data 
    if state == 'train':
        global_feat = torch.from_numpy(train_global_feat)
    elif state == 'val' or state == 'test' or state =='test_realdata':
        global_feat = torch.from_numpy(eval_global_feat)
        
    num_fiber = points.shape[0]
    num_point_per_fiber = points.shape[1]
    # label
    if state != 'test_realdata':
        label = label[:,0]  # [B,1] to [B]   
    # points
    points = points.transpose(2, 1)  # points [B, 3, N_point]
    # local feat
    klocal_feat_set = klocal_feat_set.transpose(2,1)  # [B,3,N_point,k]
    # global feat
    if state == 'test_realdata': 
        kglobal_point_set = global_feat.repeat(num_fiber,1,1,1).transpose(2,1)  # [B,3,N_point,k_global]
        new_subidx=torch.zeros(num_fiber).long() 
    else:
        new_subidx = new_subidx[:,0]  # [B,1] to [B]
        kglobal_point_set = global_feat.transpose(2,1)  # [num_subject*num_aug,3,N_point,k_global]
        kglobal_point_set = kglobal_point_set[new_subidx, ...]  # [B,3,N_point,k_global].
    # concat knn and random feat to get info feat
    if args.k == 0 and args.k_global == 0:
        info_point_set = torch.Tensor([0])
    elif args.k == 0 and args.k_global > 0:
        info_point_set = kglobal_point_set
    elif args.k > 0 and args.k_global == 0:
        info_point_set = klocal_feat_set
    elif args.k > 0 and args.k_global > 0:            
        info_point_set = torch.cat((klocal_feat_set, kglobal_point_set), dim=3)  # [B,3,N_point,k+k_global]
    else:
        raise ValueError('Invalid k and k sparse values')
    if (args.k>0 or args.k_global>0) and (idx_data==0 and epoch==1):
        if state != 'test_realdata': 
            start_idx = 0
            end_idx = 100
            org_info_feat_save_folder = os.path.join(args.out_path,'org_info_feat_vtk',state)
            makepath(org_info_feat_save_folder)
            save_info_feat(points, info_point_set, new_subidx, start_idx, end_idx, args.aug_times, args.k, 
                        args.k_global, args.k_ds_rate, org_info_feat_save_folder)
    if state == 'test_realdata':
        points, info_point_set = points.to(device), info_point_set.to(device)
    else:
        points, label, info_point_set = points.to(device), label.to(device), info_point_set.to(device)
    
    if state == 'train':
        optimizer.zero_grad()
        net = net.train()
    else:
        net = net.eval() 
    # get desired results for pred -- [B,N_point,Cls] for seg, [B,Cls] for cls
    if args.model_name == 'dgcnn':
        pred = net(points, info_point_set)   
    elif args.model_name == 'pointnet':
        pred,_,_=net(points, info_point_set)
    else:
        raise ValueError('Please input valid model name dgcnn | pointnet')
    pred = pred.view(-1, num_classes)  # seg (B,N_point,Cls) -> (B*N_point,Cls); cls (B,Cls) -> (B,Cls)
    
    if state != 'test_realdata':
        label = label.view(-1,1)[:,0]      # seg (B*N_point); cls (B)
        loss = F.nll_loss(pred, label)
    if state == 'train':
        loss.backward()
        optimizer.step()
        if args.scheduler == 'wucd':
            scheduler.step(epoch-1 + idx_data/num_batch)
    
    _, pred_idx = torch.max(pred, dim=1)    # (B*N_point,Cls)->(B*N_point,) for seg, (B,Cls)-> (B) for cls         
            
    if state != 'test_realdata':
        total_loss += loss.item()
        # for calculating weighted and macro metrics
        label = label.cpu().detach().numpy()
        labels_lst.extend(label)
    pred_idx = pred_idx.cpu().detach().numpy()
    predicted_lst.extend(pred_idx)
    
    return total_loss, labels_lst, predicted_lst
    
    
def train_val_DL_net(net):
    """train and validation of the network"""
    time_start = time.time()
    train_num_batch = train_data_size / args.train_batch_size
    val_num_batch = val_data_size / args.val_batch_size
    # save training and validating process data
    train_loss_lst, val_loss_lst, train_acc_lst, val_acc_lst, \
    train_precision_lst, val_precision_lst, train_recall_lst, val_recall_lst, \
    train_f1_lst, val_f1_lst = [], [], [], [], [], [], [], [], [], []
    # save weights with best metrics
    org_best_f1_mac, tract_best_f1_mac = 0,0   
    org_best_f1_epoch, tract_best_f1_epoch = 1,1
    org_best_f1_wts, tract_best_f1_wts = None,None
    org_best_f1_val_labels_lst, tract_best_f1_val_labels_lst = [],[]
    org_best_f1_val_pred_lst,tract_best_f1_val_pred_lst = [],[]
    
    for epoch in range(args.epoch):
        train_start_time = time.time()
        epoch += 1
        total_train_loss, total_val_loss = 0, 0
        train_labels_lst, train_predicted_lst = [], []
        val_labels_lst, val_predicted_lst = [], []
        # training
        for i, data in enumerate(train_loader, start=0):
            total_train_loss, train_labels_lst, train_predicted_lst = \
                train_val_test_forward(i, data, net, 'train', total_train_loss, train_labels_lst, train_predicted_lst, 
                                       args, device, num_classes, epoch, train_num_batch, train_global_feat=train_global_feat)
                
        if args.scheduler == 'step':
            scheduler.step()
        # train metric calculation
        train_end_time = time.time()
        train_time = round(train_end_time-train_start_time, 2)
        train_loss_lst, train_acc_lst, train_precision_lst, train_recall_lst, train_f1_lst,_,_,_,_,_,_ = \
            meters(epoch, train_num_batch, total_train_loss, train_labels_lst, train_predicted_lst, 
                   train_loss_lst, train_acc_lst, train_precision_lst, train_recall_lst, train_f1_lst, train_time, 'train')

        # validation
        with torch.no_grad():
            val_start_time = time.time()
            for j, data in enumerate(val_loader, start=0):
                total_val_loss, val_labels_lst, val_predicted_lst = \
                    train_val_test_forward(j, data, net, 'val', total_val_loss, val_labels_lst, val_predicted_lst, 
                                           args, device, num_classes, epoch, eval_global_feat=val_global_feat)
        
        # save weights regularly
        if epoch % args.save_step == 0:
            torch.save(net.state_dict(), '{}/epoch_{}_model.pth'.format(args.out_path, epoch))
            print('Save {}/epoch_{}_model.pth'.format(args.out_path, epoch))  
        
        # validation metric calculation
        val_end_time = time.time()
        val_time = round(val_end_time-val_start_time, 2)
        val_loss_lst, val_acc_lst, val_precision_lst, val_recall_lst, val_f1_lst, _, org_mac_val_f1, \
            _, tract_mac_val_f1, tract_labels_lst, tract_pred_lst = \
            meters(epoch, val_num_batch, total_val_loss, val_labels_lst, val_predicted_lst, 
                   val_loss_lst, val_acc_lst, val_precision_lst, val_recall_lst, val_f1_lst, val_time, 'val')
        # swap and save the best metric
        if org_mac_val_f1 > org_best_f1_mac:
            org_best_f1_mac, org_best_f1_epoch, org_best_f1_wts, org_best_f1_val_labels_lst, org_best_f1_val_pred_lst = \
                best_swap(org_mac_val_f1, epoch, net, val_labels_lst, val_predicted_lst)
        if tract_mac_val_f1 > tract_best_f1_mac:
            tract_best_f1_mac, tract_best_f1_epoch, tract_best_f1_wts, tract_best_f1_val_labels_lst, tract_best_f1_val_pred_lst = \
                best_swap(tract_mac_val_f1, epoch, net, tract_labels_lst, tract_pred_lst)
                
    # save best weights
    save_best_weights(net, org_best_f1_wts, args.out_path, 'org_f1', org_best_f1_epoch, org_best_f1_mac, logger)
    save_best_weights(net, tract_best_f1_wts, args.out_path, 'tract_f1', tract_best_f1_epoch, tract_best_f1_mac, logger)
        
    # plot process curves
    process_curves(args.epoch, train_loss_lst, val_loss_lst, train_acc_lst, val_acc_lst,
                   train_precision_lst, val_precision_lst, train_recall_lst, val_recall_lst,
                    train_f1_lst, val_f1_lst, -1, -1, org_best_f1_mac, org_best_f1_epoch, args.out_path)

      
    # remove checkpoints
    saved_steps = list(range(args.save_step, args.epoch+1, args.save_step))[:-1] # save the last epoch
    for epoch in saved_steps:
        os.remove('{}/epoch_{}_model.pth'.format(args.out_path, epoch))
        print('Remove {}/epoch_{}_model.pth'.format(args.out_path, epoch))
    
    # total processing time
    time_end = time.time() 
    total_time = round(time_end-time_start, 2)
    logger.info('Total processing time is {}s'.format(total_time))


def meters(epoch, num_batch, total_loss, labels_lst, predicted_lst, 
           org_loss_lst, org_acc_lst, org_precision_lst, org_recall_lst, org_f1_lst, run_time, state):
    # train accuracy loss
    avg_loss = total_loss / float(num_batch)
    org_loss_lst.append(avg_loss)
    # train macro p, r, f1
    # original labels 
    org_acc, org_mac_precision, org_mac_recall, org_mac_f1 = \
        calculate_acc_prec_recall_f1(labels_lst, predicted_lst)
    org_acc_lst.append(org_acc)
    org_precision_lst.append(org_mac_precision)
    org_recall_lst.append(org_mac_recall)
    org_f1_lst.append(org_mac_f1)
    # Tract labels
    tract_labels_lst = cluster2tract_label(labels_lst, ordered_tract_cluster_mapping_dict)
    tract_pred_lst = cluster2tract_label(predicted_lst, ordered_tract_cluster_mapping_dict)
    tract_acc, _, _, tract_mac_f1 = calculate_acc_prec_recall_f1(tract_labels_lst, tract_pred_lst)   
    
    logger.info('epoch [{}/{}] time: {}s {} loss: {} org (800clusters+800outliers) acc: {},f1: {}; Tract acc: {},f1: {}'
                 .format(epoch, args.epoch, run_time, state, round(avg_loss, 4), round(org_acc, 4), round(org_mac_f1, 4), round(tract_acc, 4), round(tract_mac_f1, 4)))
        
    return org_loss_lst, org_acc_lst, org_precision_lst, org_recall_lst, org_f1_lst, org_acc, org_mac_f1, \
           tract_acc, tract_mac_f1, tract_labels_lst, tract_pred_lst


def results_logging(args, logger, eval_state, label_names, org_labels_lst, org_predicted_lst,
                    tract_labels_lst, tract_predicted_lst, ordered_tract_cluster_mapping_dict):
    """log results for original (800 clusters + 800 outliers) labels and tract (42+1other) labels"""
    if args.use_tracts_training:   # if use tracts as training data, then use tracts as testing data
        assert args.use_tracts_testing == True
        
    # calculate classification report and plot class analysis curves for different metrics
    if not args.use_tracts_training:
        label_names_str = label_names
        # best metric
        h5_name = 'unrelatedHCP_{}_results_best{}.h5'.format(eval_state, args.best_metric)
        try:
            logger.info('{} original labels classification report as below'.format(len(label_names_str)))
            classify_report(org_labels_lst, org_predicted_lst, label_names_str, logger, args.out_log_path, args.best_metric, 
                            eval_state, h5_name, obtain_conf_mat=False)
            org_label_best_acc,_,_,org_label_best_mac_f1 = calculate_acc_prec_recall_f1(org_labels_lst, org_predicted_lst)
        except:
            logger.info('Warning!! Number of classes, {}, does not match size of target_names, {}. Try specifying the labels parameter'
                        .format(np.unique(np.array(org_predicted_lst)).shape[0], len(label_names_str)))
            
    if args.use_tracts_testing:
        tract_label_names_str = list(ordered_tract_cluster_mapping_dict.keys())
        h5_name = 'unrelatedHCP_{}_results_TractLabels_best{}.h5'.format(eval_state, args.best_metric)
        try:
            logger.info('{}+1 tract labels classification report as below'.format(len(tract_label_names_str)-1))
            classify_report(tract_labels_lst, tract_predicted_lst, tract_label_names_str, logger, args.out_log_path, 
                        args.best_metric, eval_state, h5_name, obtain_conf_mat=True)
            tract_label_best_acc,_,_,tract_label_best_mac_f1 = calculate_acc_prec_recall_f1(tract_labels_lst, tract_predicted_lst)
        except:
            logger.info('Warning!! Number of classes, {}, does not match size of target_names, {}. Try specifying the labels parameter'
                         .format(np.unique(np.array(tract_predicted_lst)).shape[0], len(tract_label_names_str))) 
    
    # accuracy, f1    
    try:
        logger.info("Results for {} original labels with best f1 weights: Acc {} F1 {}".format(len(label_names_str), round_decimal(org_label_best_acc,5),round_decimal(org_label_best_mac_f1,5)))
    except:
        pass
    try:
        logger.info("Results for {}+1 tract labels with best f1 weights: Acc {} F1 {}".format(len(tract_label_names_str)-1, round_decimal(tract_label_best_acc,5),round_decimal(tract_label_best_mac_f1,5)))        
    except:
        pass 
    

def train_val_paths():
    """paths"""
    args.input_path = unify_path(args.input_path)
    args.out_path_base = unify_path(args.out_path_base)
    # Train and validation
    args.out_path = os.path.join(args.out_path_base)
    makepath(args.out_path)


if __name__ == '__main__':
    # GPU check
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Variable Space
    parser = create_parser()
    args = parser.parse_args()
    # fix seed
    args.manualSeed = 0 
    print("Random Seed: ", args.manualSeed)
    fix_seed(args.manualSeed)
    # adaptively change the args
    args = adaptive_args(args)
    # convert str to num
    args.rot_ang_lst = str2num(args.rot_ang_lst)
    args.scale_ratio_range = str2num(args.scale_ratio_range)
    # save local+global feature
    args.save_knn_neighbors = True
    # paths
    train_val_paths()
    # Tract cluster mapping
    ordered_tract_cluster_mapping_dict = obtain_TractClusterMapping()  # {'tract name': ['cluster_xxx','cluster_xxx', ... 'cluster_xxx']}
    # Record the training process and values
    logger = create_logger(args.out_path)
    logger.info('=' * 55)
    logger.info(args)
    logger.info('=' * 55)
    if not args.save_args_only:
        # load data
        train_loader, val_loader, label_names, num_classes, train_data_size, val_data_size, eval_state, train_global_feat, val_global_feat \
            = load_batch_data()
        # model setting
        DL_model = load_model(args, num_classes, device)
        optimizer, scheduler = load_settings(DL_model)
        # train and eval net
        train_val_DL_net(DL_model)
    # save args
    args_path = os.path.join(args.out_path, 'cli_args.txt')
    save_args(args_path, args)