import whitematteranalysis as wma
import argparse
import os
import time

import torch
import torch.nn.parallel

import sys
sys.path.append('../')

from utils.logger import create_logger
from utils.funcs import cluster2tract_label, makepath,  obtain_TractClusterMapping,tractography_parcellation
from utils.cli import load_args_in_testing_only
from train import load_model, train_val_test_forward
from datasets.dataset import RealData_PatchData, center_tractography
import utils.tract_feat as tract_feat


def test_realdata_DL_net(net):
    """test the network"""
    test_labels_lst, test_predicted_lst = [], []
    # test
    with torch.no_grad():
        for j, data in enumerate(test_loader, start=0):
            _, _, test_predicted_lst = \
                train_val_test_forward(j, data, net, 'test_realdata', -1, [], test_predicted_lst,
                                       args, device, args.num_classes, epoch=1, eval_global_feat=test_realdata_global_feat)

    return test_labels_lst, test_predicted_lst


start_time = time.time()
use_cpu = False
if use_cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")

# Parse arguments
parser = argparse.ArgumentParser(description="Test on real data", epilog="Referenced from https://github.com/SlicerDMRI/SupWMA"
                                             "Tengfei Xue txue4133@uni.sydney.edu.au")
# paths
parser.add_argument('--input_path', type=str, default='../TrainData_800clu800ol',
                    help='The input path for train/val/test (atlas) data')
parser.add_argument('--weight_path_base', type=str, help='pretrained network model')
parser.add_argument('--tractography_path', type=str, help='Tractography data as a vtkPolyData file')
parser.add_argument('--out_path', type=str, help='The output directory can be a new empty directory. It will be created if needed.')
# model parameters
parser.add_argument('--model_name', type=str, default='dgcnn', help='The name of the point cloud model')
parser.add_argument('--k', type=int, default=20, help='the number of neighbor points (in streamline level)')
parser.add_argument('--k_global', type=int, default=80, help='The number of points (in streamline level) for the random sparse sampling')
parser.add_argument('--k_point_level', type=int, default=5, help='the number of neighbor points (in point level)')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',help='Dimension of embeddings')
parser.add_argument('--dropout', type=float, default=0.5, help='initial dropout rate')
# process parameters
parser.add_argument('--test_realdata_batch_size', type=int, default=6144, help='batch size')
parser.add_argument('--num_fiber_per_brain', type=int, default=10000, help='number of fibers for each brain, keep consistent with the training data')
parser.add_argument('--num_points', type=int, default=15, help='number of points on each fiber in the tractography')
parser.add_argument('--num_classes', type=int, default=1600, help='number of classes')
parser.add_argument('--cal_equiv_dist', default=False, action='store_true', help='Calculate equivalent distance for pairwise distance matrix for finding neighbors')
parser.add_argument('--k_ds_rate', type=float, default=0.1, help='downsample the tractography when calculating pairwise distance matrix')


args = parser.parse_args()
# input from test_realdata.py keyboard
args_path = args.weight_path_base + '/cli_args.txt'
# input from train.py keyboard, in cli.txt
args = load_args_in_testing_only(args_path, args)
# paths
args.weight_path = os.path.join(args.weight_path_base, 'best_tract_f1_model.pth')
makepath(args.out_path)
# create logger
log_path = os.path.join(args.out_path, 'log')
makepath(log_path)
logger = create_logger(log_path)
logger.info('=' * 55)
logger.info(args)
logger.info('=' * 55)
# label names
ordered_tract_cluster_mapping_dict = obtain_TractClusterMapping()
tract_label_names_str = list(ordered_tract_cluster_mapping_dict.keys())
# read tractography into feature
pd_tractography = wma.io.read_polydata(args.tractography_path)
logger.info('Finish reading tractography from: {}'.format(args.tractography_path))
tractography_file_n = os.path.basename(os.path.normpath(args.tractography_path))
# load non-registered feature 
logger.info('Extracting feature from unregistered tractography'.format(args.tractography_path))
feat_RAS, _ = tract_feat.feat_RAS(pd_tractography, number_of_points=args.num_points)
logger.info('The number of fibers in test tractography is {}'.format(feat_RAS.shape[0]))
logger.info('Extracting RAS features is done')
# re-center tractography
save_recentered_data = False
centered_feat_RAS = center_tractography(args.input_path,feat_RAS,args.out_path, 
                                        logger,tractography_file_n,save_recentered_data)
logger.info('re-centering tractography is done.')
# Real data processing
test_realdata = RealData_PatchData(centered_feat_RAS, k=args.k, k_global=args.k_global, cal_equiv_dist=args.cal_equiv_dist, 
                                    use_endpoints_dist=False, rough_num_fiber_each_iter=args.num_fiber_per_brain, k_ds_rate=args.k_ds_rate) 
test_loader = torch.utils.data.DataLoader(test_realdata, batch_size=args.test_realdata_batch_size, shuffle=False)
test_realdata_global_feat = test_realdata.global_feat
test_realdata_size = len(test_realdata)
logger.info('calculating knn+random features is done')
# test network
DL_model = load_model(args, num_classes=args.num_classes, device=device, test=True)  
_, predicted_lst = test_realdata_DL_net(DL_model)

tract_predicted_lst = cluster2tract_label(predicted_lst, ordered_tract_cluster_mapping_dict)
logger.info('Deep learning prediction is done.')
# save parcellated vtp results
tractography_parcellation(args, pd_tractography, tract_predicted_lst, tract_label_names_str)  # pd tractography here is in the subject space.
# time
end_time = time.time()
tot_time = round(end_time - start_time, 3)
logger.info('All done!!! Total time is {}s'.format(tot_time))
    