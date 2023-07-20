import argparse
import json
import math
from argparse import Namespace

def create_parser():
    # Variable Space
    parser = argparse.ArgumentParser(description="Train and evaluate a model",
                                     epilog="by Tengfei Xue txue4133@uni.sydney.edu.au")
    # Paths
    parser.add_argument('--input_path', type=str, default='./TrainData/outliers_data/DEBUG_kp0.1/h5_np15/',
                        help='Input graph data and labels')
    parser.add_argument('--out_path_base', type=str, default='./ModelWeights', help='Save trained models')
    # Data augmentation parameters
    parser.add_argument('--rot_ang_lst', type=str, default="0 0 0", help='"45 15 15" rotate 45 degrees in LR(x) axis, 15 degrees in AP(y) axis, 15 degrees in SI(z) axis')
    parser.add_argument('--scale_ratio_range', type=str, default="0 0", help='random scale between [1+scale_ratio_range[0], 1+scale_ratio_range[1]]; -0.35 0.05 means scale between 0.65 and 1.05')
    parser.add_argument('--trans_dis', type=float, default=0.0, help='random translation between [-trans_dis, +trans_dis]')
    parser.add_argument('--aug_times', type=int, default=10, help='How many augmented data we will get for each data.')
    # Local-global representations
    parser.add_argument('--k', type=int, default=20, help='Local streamlines (k_local).the number of neighbor streamlines (in streamline level)')
    parser.add_argument('--k_ds_rate', type=float, default=0.1, help='1 means no downsample. downsample the tractography when calculating pairwise distance matrix for local streamlines.')
    parser.add_argument('--k_global', type=int, default=500, help='Global streamlines (k_global). The number of streamlines (in streamline level) for random sampling')
    parser.add_argument('--k_point_level', type=int, default=5, help='The number of neighbor points (in point level) on one streamline')
    # Training parameters
    parser.add_argument('--save_step', type=int, default=5, help='The interval of saving weights')
    parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',help='Dimension of embeddings')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--opt', type=str, default='Adam', help='type of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for Adam')
    parser.add_argument('--momentum', type=float, default=0, help='momentum for SGD')
    parser.add_argument('--scheduler', type=str, default='step', help='type of learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=10, help='Period of learning rate decay')
    parser.add_argument('--decay_factor', type=float, default=0, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--T_0', type=int, default=10, help='Number of iterations for the first restart (for wucd)')
    parser.add_argument('--T_mult', type=int, default=2, help='A factor increases Ti after a restart (for wucd)')
    parser.add_argument('--dropout', type=float, default=0.5, help='initial dropout rate')
    parser.add_argument('--train_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--val_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='the number of epochs')
    parser.add_argument('--best_metric', type=str, default='f1', help='evaluation metric')
    parser.add_argument('--model_name', type=str, default='dgcnn', help='The name of the point cloud model')
    parser.add_argument('--num_fiber_per_brain', type=int, default=1000, help='The number of fibers each brain')
    parser.add_argument('--num_point_per_fiber', type=int, default=15, help='The number of points each fiber')
    parser.add_argument('--use_tracts_training', default=False, action='store_true', help='Convert cluster labels into tracts during training')
    parser.add_argument('--use_tracts_testing', default=False, action='store_true', help='Convert cluster labels into tracts during testing')
    parser.add_argument('--save_args_only', default=False, action='store_true', help='Save args only, not perform training')
    parser.add_argument('--cal_equiv_dist', default=False, action='store_true', help='Calculate equivalent distance for pairwise distance matrix')
    parser.add_argument('--recenter', default=False, action='store_true', help='Recenter the data use the center of mass')
    parser.add_argument('--include_org_data', default=False, action='store_true', help='Include original data when augmenting data')
    
    return parser


def adaptive_args(args):
    if args.trans_dis == 0:
        args.recenter = False
        
    return args


def load_args(path, args):
    params_set_in_testing = ['aug_times', 'out_path']  # For the parameter name in this list, input the parameter value from test.py or test_realdata.py
    with open(path, 'r') as f:
        saved_json_dict = json.load(f)
        args_dict = vars(args)
        for key,value in saved_json_dict.items():
            if key in params_set_in_testing:
                print('Skip loading {} from training args'.format(key))
                continue
            args_dict[key] = value
        args = Namespace(**args_dict)
    return args


def load_args_in_testing_only(path, args):
    """Only load augments that are used in testing"""
    params_set_in_testing = ['aug_times', 'out_path']  # For the parameter name in this list, input the parameter value from test.py or test_realdata.py
    with open(path, 'r') as f:
        saved_json_dict = json.load(f)
        args_dict = vars(args)
        for key,value in saved_json_dict.items():
            if key in params_set_in_testing:
                print('Skip loading {} from training args'.format(key))
                continue
            if key in args_dict.keys():  # only if arguments also appear at testing, we then load them from training.
                args_dict[key] = value
        args = Namespace(**args_dict)
    return args


def save_args(path, args):
    with open(path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print(args.__dict__)