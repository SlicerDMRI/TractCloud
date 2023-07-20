import os
import sys
sys.path.append('../')
import time
import numpy as np

import torch
import torch.nn.parallel
import torch.utils.data

from utils.logger import create_logger
from utils.funcs import cluster2tract_label, unify_path, makepath, fix_seed, obtain_TractClusterMapping
from utils.cli import create_parser, load_args, adaptive_args
from train import load_datasets, load_model, results_logging, train_val_test_forward


def load_batch_data():
    """load test data"""
    eval_state = 'test'
    _, test_dataset = load_datasets(eval_split=eval_state, args=args, test=True, logger=logger)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size,
        shuffle=True)

    test_data_size = len(test_dataset)
    try:
        test_feat_shape = test_dataset.fiber_feat.shape
    except:
        test_feat_shape = test_dataset.org_feat.shape
    logger.info('The testing data feature size is:{}'.format(test_feat_shape))

    # load label names
    if args.use_tracts_training:
        label_names =  list(ordered_tract_cluster_mapping_dict.keys()) 
    else:
        label_names = test_dataset.label_names
    num_classes = len(np.unique(label_names))
    logger.info('The number of classes is:{}'.format(num_classes))

    # test random feat
    test_global_feat = test_dataset.global_feat

    return test_loader, label_names, num_classes, test_data_size, eval_state, test_global_feat 


def test_DL_net(net):
    """test the network"""
    total_test_loss = 0
    test_labels_lst, test_predicted_lst = [], []
    # test
    with torch.no_grad():
        for j, data in enumerate(test_loader, start=0):
            total_test_loss, test_labels_lst, test_predicted_lst = \
                train_val_test_forward(j, data, net, 'test', total_test_loss, test_labels_lst, test_predicted_lst, 
                                       args, device, num_classes, epoch=1, eval_global_feat=test_global_feat)
    
    return test_labels_lst, test_predicted_lst
    
    
def test_paths():
    # paths
    args.input_path = unify_path(args.input_path)
    args.out_path_base = unify_path(args.out_path_base)
    args.out_path = os.path.join(args.out_path_base)
    # test
    if args.aug_times >0:
        out_log_path_base = os.path.join(args.out_path, 'log_AugTimes{}'.format(args.aug_times))
    else:
        out_log_path_base = os.path.join(args.out_path, 'log_NoAug')
    args.out_log_path = os.path.join(out_log_path_base)
    makepath(args.out_log_path)


if __name__ == '__main__':
    # GPU check
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Variable Space
    parser = create_parser()
    args = parser.parse_args()
    # input from test.py keyboard
    args_path = args.out_path_base + '/cli_args.txt'
    # input from train.py keyboard, in cli.txt
    args = load_args(args_path, args)
    # fix seed
    fix_seed(args.manualSeed)
    # adaptively change the args
    args = adaptive_args(args)
    # paths
    test_paths()
    # Tract cluster mapping
    ordered_tract_cluster_mapping_dict = obtain_TractClusterMapping()  # {'tract name': ['cluster_xxx','cluster_xxx', ... 'cluster_xxx']} 
    # Record the training process and values
    logger = create_logger(args.out_log_path)
    logger.info('=' * 55)
    logger.info(args)
    logger.info('=' * 55)
    # load data
    test_loader, label_names, num_classes, test_data_size, eval_state, test_global_feat = load_batch_data()
    
    # Test on best org and tract weights separately
    time_start = time.time()
    weight_prefix_lst = ['org', 'tract']
    for weight_prefix in weight_prefix_lst:  # load best weight one by one
        # model setting
        args.weight_path = os.path.join(args.out_path, 'best_{}_{}_model.pth'.format(weight_prefix, args.best_metric))
        logger.info("Load best {} {} model".format(weight_prefix, args.best_metric))
        DL_model = load_model(args, num_classes, device, test=True)
        # test net
        if weight_prefix == 'org':
            org_labels_lst, org_predicted_lst = test_DL_net(DL_model)
        elif weight_prefix == 'tract':
            labels_lst, predicted_lst = test_DL_net(DL_model)
            tract_labels_lst = cluster2tract_label(labels_lst, ordered_tract_cluster_mapping_dict)
            tract_predicted_lst = cluster2tract_label(predicted_lst, ordered_tract_cluster_mapping_dict)
    
    # log results
    results_logging(args, logger, eval_state, label_names, org_labels_lst, org_predicted_lst, 
                    tract_labels_lst, tract_predicted_lst, ordered_tract_cluster_mapping_dict)

    # total processing time
    time_end = time.time() 
    total_time = round(time_end-time_start, 2)
    logger.info('Test on {}'.format(args.out_path))
    logger.info('Total processing time is {}s'.format(total_time))