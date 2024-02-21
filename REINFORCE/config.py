import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
# from tensorboardX import SummaryWriter
import os
import json
import numpy as np
import sys
# sys.path.append('../DropEdge/')
# sys.path.append('../DropEdge/src')
# from DropEdge.src.models import *
# from DropEdge.src.sample import Sampler
# from DropEdge.src.earlystopping import EarlyStopping


# sys.path.append('../pyGAT/')
# from pyGAT.models_GAT import GAT
# from pyGAT.utils_GAT import load_data, accuracy

# sys.path.append('../gmixup')
# import os.path as osp
# import torch
# import torch.nn.functional as F
# from torch_geometric.datasets import TUDataset,Planetoid
# from torch_geometric.data import DataLoader
# from torch_geometric.utils import degree
# from torch.autograd import Variable

import random
# from torch.optim.lr_scheduler import StepLR
# from gmixup.src.utils import stat_graph, split_class_graphs, align_graphs
# from gmixup.src.utils import two_graphons_mixup, universal_svd
# from gmixup.src.graphon_estimator import universal_svd
# from gmixup.src.models import GIN

import argparse
# from src.models import *

# sys.path.append('../KGE_HAKE')
# import logging
# from torch.utils.data import DataLoader
# from KGE_HAKE.codes.models import *
# from KGE_HAKE.codes.data import TrainDataset, BatchType, ModeType, DataReader
# from KGE_HAKE.codes.data import BidirectionalOneShotIterator
# from KGE_HAKE.codes.runs import override_config, save_model, set_logger, log_metrics, HAKE_training_and_test

# sys.path.append('../KGE_DURA')
# from torch import optim
# import importlib
# from KGE_DURA.code.datasets import Dataset
# from KGE_DURA.code.models import *
# from KGE_DURA.code.regularizers import *
# from KGE_DURA.code.optimizers import KBCOptimizer



arg_lists = []
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--network_type', type=str, choices=['lstm'], default='lstm')
net_arg.add_argument('--llm_task', type=str, choices=['qa'], default='qa')
net_arg.add_argument('--graph_task', type=str, choices=['kg', 'nc', 'gc', 'kge'], default='kg')


# agent
net_arg.add_argument('--tie_weights', type=str2bool, default=True)
net_arg.add_argument('--agent_dim', type=int, default=200)

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='ptb')
data_arg.add_argument('--datasetname', type=str, default='FB15K237')
data_arg.add_argument('--LP_model',type=str,choices=['transe','transh','transd','distmult','rescal','simple','complEx','analogy'],default='transe')
data_arg.add_argument('--batch_size', type=int, default= 64)
data_arg.add_argument('--train_times',type=int,default = 1000)
data_arg.add_argument('--method',type=str,choices=['scratch','finetune'],default='scratch')

# #KGE_HAKE
# kge_arg = add_argument_group('KGE')
# kge_arg.add_argument('--kge', type=str, default='DURA', choices=['DURA','HAKE'])
# kge_arg.add_argument(
#     '--HAKE_dataset', choices=['WN18', 'WN18RR', 'FB15K', 'FB15K237'],
#     help="Dataset in {}".format(['WN18', 'WN18RR', 'FB15K', 'FB15K237'])
# )
# kge_arg.add_argument('--HAKE_data_path',type=str,default='../KGE_HAKE/data')
# kge_arg.add_argument('--HAKE_do_train', action='store_true')
# kge_arg.add_argument('--HAKE_do_valid', action='store_true')
# kge_arg.add_argument('--HAKE_do_test', action='store_true')

# # kge_arg.add_argument('--HAKE_data_path', type=str, default=None)
# kge_arg.add_argument('--HAKE_model', default='TransE', type=str)

# kge_arg.add_argument('-HAKE_n', '--HAKE_negative_sample_size', default=128, type=int)
# kge_arg.add_argument('-HAKE_d', '--HAKE_hidden_dim', default=500, type=int)
# kge_arg.add_argument('-HAKE_g', '--HAKE_gamma', default=12.0, type=float)
# kge_arg.add_argument('-HAKE_a', '--HAKE_adversarial_temperature', default=1.0, type=float)
# kge_arg.add_argument('-HAKE_b', '--HAKE_batch_size', default=1024, type=int)
# kge_arg.add_argument('--HAKE_test_batch_size', default=4, type=int, help='valid/test batch size')
# kge_arg.add_argument('-HAKE_mw', '--HAKE_modulus_weight', default=1.0, type=float)
# kge_arg.add_argument('-HAKE_pw', '--HAKE_phase_weight', default=0.5, type=float)

# kge_arg.add_argument('-HAKE_lr', '--HAKE_learning_rate', default=0.0001, type=float)
# kge_arg.add_argument('-HAKE_cpu', '--HAKE_cpu_num', default=10, type=int)
# kge_arg.add_argument('-HAKE_init', '--HAKE_init_checkpoint', default=None, type=str)
# kge_arg.add_argument('-HAKE_save', '--HAKE_save_path', default='../KGE_HAKE/data', type=str)
# kge_arg.add_argument('--HAKE_max_steps', default=100000, type=int)

# kge_arg.add_argument('--HAKE_save_checkpoint_steps', default=10000, type=int)
# kge_arg.add_argument('--HAKE_valid_steps', default=10000, type=int)
# kge_arg.add_argument('--HAKE_log_steps', default=100, type=int, help='train log every xx steps')
# kge_arg.add_argument('--HAKE_test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

# kge_arg.add_argument('--HAKE_no_decay', action='store_true', help='Learning rate do not decay')

# # DURA_datasets = ['WN18', 'WN18RR', 'FB15', 'FB237']


# kge_arg.add_argument(
#     '--DURA_dataset', choices=['WN18', 'WN18RR', 'FB15K', 'FB15K237'],
#     help="Dataset in {}".format(['WN18', 'WN18RR', 'FB15K', 'FB15K237'])
# )

# kge_arg.add_argument(
#     '--DURA_model', type=str, default='CP'
# )

# kge_arg.add_argument(
#     '--DURA_regularizer', type=str, default='NA',
# )

# # DURA_optimizers = ['Adagrad', 'Adam', 'SGD']
# kge_arg.add_argument(
#     '--DURA_optimizer', choices=['Adagrad', 'Adam', 'SGD'], default='Adagrad',
#     help="Optimizer in {}".format(['Adagrad', 'Adam', 'SGD'])
# )

# kge_arg.add_argument(
#     '--DURA_max_epochs', default=50, type=int,
#     help="Number of epochs."
# )
# kge_arg.add_argument(
#     '--DURA_valid', default=3, type=float,
#     help="Number of epochs before valid."
# )
# kge_arg.add_argument(
#     '--DURA_rank', default=1000, type=int,
#     help="Factorization rank."
# )
# kge_arg.add_argument(
#     '--DURA_batch_size', default=1000, type=int,
#     help="Factorization rank."
# )
# kge_arg.add_argument(
#     '--DURA_reg', default=0, type=float,
#     help="Regularization weight"
# )
# kge_arg.add_argument(
#     '--DURA_init', default=1e-3, type=float,
#     help="Initial scale"
# )
# kge_arg.add_argument(
#     '--DURA_learning_rate', default=1e-1, type=float,
#     help="Learning rate"
# )
# kge_arg.add_argument(
#     '--DURA_decay1', default=0.9, type=float,
#     help="decay rate for the first moment estimate in Adam"
# )
# kge_arg.add_argument(
#     '--DURA_decay2', default=0.999, type=float,
#     help="decay rate for second moment estimate in Adam"
# )

# kge_arg.add_argument('-DURA_train', '--DURA_do_train', action='store_true')
# kge_arg.add_argument('-DURA_test', '--DURA_do_test', action='store_true')
# kge_arg.add_argument('-DURA_save', '--DURA_do_save', action='store_true')
# kge_arg.add_argument('-DURA_weight', '--DURA_do_ce_weight', action='store_true')
# kge_arg.add_argument('-DURA_path', '--DURA_save_path', type=str, default='../KGE_DURA/logs/')
# kge_arg.add_argument('-DURA_id', '--DURA_model_id', type=str, default='0')
# kge_arg.add_argument('-DURA_ckpt', '--DURA_checkpoint', type=str, default='')

# agent
learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test'])
learn_arg.add_argument('--reward_c', type=int, default=80,
                       help="WE DON'T KNOW WHAT THIS VALUE SHOULD BE") # TODO
learn_arg.add_argument('--ema_baseline_decay', type=float, default=0.95)
learn_arg.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])
learn_arg.add_argument('--max_epoch', type=int, default=10)
learn_arg.add_argument('--llm_test_epoch', type=int, default=1)
learn_arg.add_argument('--agent_optim', type=str, default='adam')
learn_arg.add_argument('--agent_lr', type=float, default=3.5e-4,
                       help="will be ignored if --agent_lr_cosine=True")
learn_arg.add_argument('--entropy_coeff', type=float, default=1e-4)
learn_arg.add_argument('--softmax_temperature', type=float, default=5.0)
learn_arg.add_argument('--tanh_c', type=float, default=2.5)
learn_arg.add_argument('--discount', type=float, default=0.8)
learn_arg.add_argument('--agent_grad_clip', type=float, default=0)
learn_arg.add_argument('--alpha',type=float,default=1.0)
learn_arg.add_argument('--beta',type=float,default=0.2)
learn_arg.add_argument('--gamma',type=float,default=0.1)
learn_arg.add_argument('--alpha1',type=float,default=10)
learn_arg.add_argument('--beta1',type=float,default=5)
learn_arg.add_argument('--gamma1',type=float,default=7)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--num_gpu', type=int, default=3)
misc_arg.add_argument('--random_seed', type=int, default=0)
misc_arg.add_argument('--log_epoch', type=int, default=1)


# # NC tasks
# nc_arg = add_argument_group('NC tasks')
# nc_arg.add_argument('--no_cuda', action='store_true', default=False,
#                         help='Disables CUDA training.')
# nc_arg.add_argument('--fastmode', action='store_true', default=False,
#                         help='Disable validation during training.')
# nc_arg.add_argument('--gcn_epochs', type=int, default=800,
#                         help='Number of epochs to train.')
# nc_arg.add_argument('--lr', type=float, default=0.02,
#                         help='Initial learning rate.')
# nc_arg.add_argument('--lradjust', action='store_true',
#                         default=False, help='Enable leraning rate adjust.(ReduceLROnPlateau or Linear Reduce)')
# nc_arg.add_argument('--weight_decay', type=float, default=5e-4,
#                         help='Weight decay (L2 loss on parameters).')
# nc_arg.add_argument("--mixmode", action="store_true",
#                         default=False, help="Enable CPU GPU mixing mode.")
# nc_arg.add_argument("--warm_start", default="",
#                         help="The model name to be loaded for warm start.")
# nc_arg.add_argument('--debug', action='store_true',
#                         default=False, help="Enable the detialed training output.")
# nc_arg.add_argument('--datapath', default="data/", help="The data path.")
# nc_arg.add_argument("--early_stopping", type=int,
#                         default=0,
#                         help="The patience of earlystopping. Do not adopt the earlystopping when it equals 0.")
# nc_arg.add_argument("--no_tensorboard", default=False, help="Disable writing logs to tensorboard")
# nc_arg.add_argument("--topk", default=3, help="top-K neighbors added using hyper-graph")

# # Model parameter
# nc_arg.add_argument('--type',
#                         help="Choose the model to be trained.(mutigcn, resgcn, densegcn, inceptiongcn)")
# nc_arg.add_argument('--inputlayer', default='gcn',
#                         help="The input layer of the model.")
# nc_arg.add_argument('--outputlayer', default='gcn',
#                         help="The output layer of the model.")
# nc_arg.add_argument('--hidden', type=int, default=128,
#                         help='Number of hidden units.')
# nc_arg.add_argument('--dropout', type=float, default=0.5,
#                         help='Dropout rate (1 - keep probability).')
# nc_arg.add_argument('--withbn', action='store_true', default=False,
#                         help='Enable Bath Norm GCN')
# nc_arg.add_argument('--withloop', action="store_true", default=False,
#                         help="Enable loop layer GCN")
# nc_arg.add_argument('--nhiddenlayer', type=int, default=1,
#                         help='The number of hidden layers.')
# nc_arg.add_argument("--normalization", default="AugNormAdj",
#                         help="The normalization on the adj matrix.")
# nc_arg.add_argument("--sampling_percent", type=float, default=1.0,
#                         help="The percent of the preserve edges. If it equals 1, no sampling is done on adj matrix.")
# # nc_arg.add_argument("--baseblock", default="res", help="The base building block (resgcn, densegcn, mutigcn, inceptiongcn).")
# nc_arg.add_argument("--nbaseblocklayer", type=int, default=1,
#                         help="The number of layers in each baseblock")
# nc_arg.add_argument("--aggrmethod", default="default",
#                         help="The aggrmethod for the layer aggreation. The options includes add and concat. Only valid in resgcn, densegcn and inecptiongcn")
# nc_arg.add_argument("--task_type", default="full",
#                         help="The node classification task type (full and semi). Only valid for cora, citeseer and pubmed dataset.")

# # NC tasks on GAT models
# gat_arg = add_argument_group('GAT tasks')
# gat_arg.add_argument('--GAT_no-cuda', action='store_true', default=False, help='Disables CUDA training.')
# gat_arg.add_argument('--GAT_fastmode', action='store_true', default=False, help='Validate during training pass.')
# gat_arg.add_argument('--GAT_sparse', action='store_true', default=False, help='GAT with sparse version or not.')
# gat_arg.add_argument('--GAT_seed', type=int, default=72, help='Random seed.')
# gat_arg.add_argument('--GAT_epochs', type=int, default=10, help='Number of epochs to train.')
# gat_arg.add_argument('--GAT_lr', type=float, default=0.005, help='Initial learning rate.')
# gat_arg.add_argument('--GAT_weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
# gat_arg.add_argument('--GAT_hidden', type=int, default=8, help='Number of hidden units.')
# gat_arg.add_argument('--GAT_nb_heads', type=int, default=8, help='Number of head attentions.')
# gat_arg.add_argument('--GAT_dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
# gat_arg.add_argument('--GAT_alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
# gat_arg.add_argument('--GAT_patience', type=int, default=100, help='Patience')


# # GC tasks
# gc_arg = add_argument_group('GC tasks')
# gc_arg.add_argument('--GC_data_path', type=str, default="./", help='Path to the data directory.')
# gc_arg.add_argument('--GC_dataset', type=str, default="REDDIT-BINARY", help='Name of the dataset.')
# gc_arg.add_argument('--GC_model', type=str, default="GIN", help='Model to be used.')
# gc_arg.add_argument('--GC_epoch', type=int, default=800, help='Number of training epochs.')
# gc_arg.add_argument('--GC_batch_size', type=int, default=32, help='Batch size for training.')
# gc_arg.add_argument('--GC_lr', type=float, default=0.01, help='Learning rate.')
# gc_arg.add_argument('--GC_num_hidden', type=int, default=64, help='Number of hidden units.')
# gc_arg.add_argument('--GC_gmixup', type=str, default="False", help='Enable or disable gmixup.')
# gc_arg.add_argument('--GC_lam_range', type=str, default="[0.005, 0.01]", help='Range for lambda in gmixup.')
# gc_arg.add_argument('--GC_aug_ratio', type=float, default=0.15, help='Data augmentation ratio.')
# gc_arg.add_argument('--GC_aug_num', type=int, default=10, help='Number of data augmentations.')
# gc_arg.add_argument('--GC_gnn', type=str, default="gin", help='Type of GNN model.')
# gc_arg.add_argument('--GC_seed', type=int, default=1314, help='Random seed.')
# gc_arg.add_argument('--GC_log_screen', type=str, default="False", help='Log output to screen.')
# gc_arg.add_argument('--GC_ge', type=str, default="MC", help='Graph embedding method.')


def get_args():

    args, unparsed = parser.parse_known_args()
    args.cuda = True if args.num_gpu > 0 else False
    args.device = 'cuda' if args.cuda else 'cpu'
    if len(unparsed) > 1:
        print(f"Unparsed args: {unparsed}")
    return args, unparsed

# def GAT_model_init(args):
#     args = parser.parse_args()
#     args.cuda = not args.no_cuda and torch.cuda.is_available()

#     random.seed(args.GAT_seed)
#     np.random.seed(args.GAT_seed)
#     torch.manual_seed(args.GAT_seed)
#     if args.cuda:
#         torch.cuda.manual_seed(args.GAT_seed)

#     # Load data
#     adj, features, labels, idx_train, idx_val, idx_test = load_data()

#     # Model and optimizer
#     model = GAT(nfeat=features.shape[1], 
#                 nhid=args.GAT_hidden, 
#                 nclass=int(labels.max()) + 1, 
#                 dropout=args.GAT_dropout, 
#                 nheads=args.GAT_nb_heads, 
#                 alpha=args.alpha)
#     optimizer = optim.Adam(model.parameters(), 
#                         lr=args.GAT_lr, 
#                         weight_decay=args.GAT_weight_decay)

#     if args.cuda:
#         model.cuda()
#         features = features.cuda()
#         adj = adj.cuda()
#         labels = labels.cuda()
#         idx_train = idx_train.cuda()
#         idx_val = idx_val.cuda()
#         idx_test = idx_test.cuda()

#     features, adj, labels = Variable(features), Variable(adj), Variable(labels)
#     args.model = model
#     args.features = features
#     args.adj = adj
#     args.idx_train = idx_train
#     args.labels = labels
#     args.optimizer = optimizer
#     #args.fastmode = args.fastmode
#     args.idx_val = idx_val
#     args.idx_test = idx_test
#     args.device = 'cuda' if args.cuda else 'cpu'
#     args.hidden = args.GAT_hidden
#     args.dropout = args.GAT_dropout
#     args.nb_heads = args.GAT_nb_heads
#     args.lr = args.GAT_lr
#     args.weight_decay = args.GAT_weight_decay



#     return args

# def KGE_HAKE_init(args):
#     if args.HAKE_model == 'HAKE':
#         args.HAKE_data_path = args.HAKE_data_path + '/' + args.HAKE_dataset
#         args.HAKE_save_path = args.HAKE_save_path + '/' + args.HAKE_model+'/' + args.HAKE_dataset
#         if (not args.HAKE_do_train) and (not args.HAKE_do_valid) and (not args.HAKE_do_test):
#             raise ValueError('one of train/val/test mode must be choosed.')

#         if args.HAKE_init_checkpoint:
#             override_config(args)
#         elif args.HAKE_data_path is None:
#             raise ValueError('one of init_checkpoint/data_path must be choosed.')
#         if args.HAKE_do_train and args.HAKE_save_path is None:
#             raise ValueError('Where do you want to save your trained model?')

#         if args.HAKE_save_path and not os.path.exists(args.HAKE_save_path):
#             os.makedirs(args.HAKE_save_path)
#         data_reader = DataReader(args.HAKE_data_path)
#         num_entity = len(data_reader.entity_dict)
#         num_relation = len(data_reader.relation_dict)
#         kge_model = HAKE(num_entity, num_relation, args.HAKE_hidden_dim, args.HAKE_gamma, args.HAKE_modulus_weight, args.HAKE_phase_weight)
#         device = 'cuda'
#         kge_model.to(device)
#         args.model = kge_model

#     # Write logs to checkpoint and console
#     set_logger(args)
#     return args

# def create_instance(module_name, class_name, *args, **kwargs):
#     module = importlib.import_module(module_name)
#     cls = getattr(module, class_name)
#     return cls(*args, **kwargs)

# def KGE_DURA_init(args):
#     if args.DURA_do_save:
#         assert args.DURA_save_path
#         save_suffix = args.DURA_model + '_' + args.DURA_regularizer + '_' + args.DURA_dataset + '_' + args.DURA_model_id

#         if not os.path.exists(args.DURA_save_path):
#             os.mkdir(args.DURA_save_path)

#         save_path = os.path.join(args.DURA_save_path, save_suffix)
#         if not os.path.exists(save_path):
#             os.mkdir(save_path)

#         with open(os.path.join(save_path, 'config.json'), 'w') as f:
#             json.dump(vars(args), f, indent=4)

#     data_path = "../KGE_DURA/data"
#     dataset = Dataset(data_path, args.DURA_dataset)
#     examples = torch.from_numpy(dataset.get_train().astype('int64'))

#     if args.DURA_do_ce_weight:
#         ce_weight = torch.Tensor(dataset.get_weight()).cuda()
#     else:
#         ce_weight = None

#     print(dataset.get_shape())

#     model = None
#     regularizer = None
#     if args.DURA_model == 'ComplEx':
#         print('model = ComplEx')
#         model = ComplEx(dataset.get_shape(), args.DURA_rank, args.DURA_init)
#     if args.DURA_regularizer == 'DURA_W':
#         regularizer = DURA_W(args.DURA_reg)
#     # exec('model = '+args.DURA_model+'(dataset.get_shape(), args.DURA_rank, args.DURA_init)')
#     # exec('regularizer = '+args.DURA_regularizer+'(args.DURA_reg)')
#     regularizer = [regularizer, N3(args.DURA_reg)]

#     device = 'cuda'
#     model.to(device)
#     for reg in regularizer:
#         reg.to(device)

#     optim_method = {
#         'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.DURA_learning_rate),
#         'Adam': lambda: optim.Adam(model.parameters(), lr=args.DURA_learning_rate, betas=(args.DURA_decay1, args.DURA_decay2)),
#         'SGD': lambda: optim.SGD(model.parameters(), lr=args.DURA_learning_rate)
#     }[args.DURA_optimizer]()

#     optimizer = KBCOptimizer(model, regularizer, optim_method, args.DURA_batch_size)

#     # 更新 args 的属性
#     args.DURA_dataset = dataset
#     args.DURA_examples = examples
#     args.DURA_ce_weight = ce_weight
#     args.DURA_model = model
#     args.DURA_regularizer = regularizer
#     args.DURA_optimizer = optimizer
#     args.DURA_save_path = save_path
#     args.model = model
#     return args

# def NC_model_init(args):

#     if args.debug:
#         print(args)
#     # pre setting
#     args.cuda = not args.no_cuda and torch.cuda.is_available()
#     args.mixmode = args.no_cuda and args.mixmode and torch.cuda.is_available()
#     if args.aggrmethod == "default":
#         if args.type == "resgcn":
#             args.aggrmethod = "add"
#         else:
#             args.aggrmethod = "concat"
#     if args.fastmode and args.early_stopping > 0:
#         args.early_stopping = 0
#         print("In the fast mode, early_stopping is not valid option. Setting early_stopping = 0.")
#     if args.type == "mutigcn":
#         print("For the multi-layer gcn model, the aggrmethod is fixed to nores and nhiddenlayers = 1.")
#         args.nhiddenlayer = 1
#         args.aggrmethod = "nores"

#     # random seed setting
#     np.random.seed(args.random_seed)
#     torch.manual_seed(args.random_seed)
#     if args.cuda or args.mixmode:
#         torch.cuda.manual_seed(args.random_seed)

#     # should we need fix random seed here?
#     sampler = Sampler(args.datasetname, args.datapath, args.task_type)

#     # get labels and indexes
#     labels, idx_train, idx_val, idx_test = sampler.get_label_and_idxes(args.cuda)
#     nfeat = sampler.nfeat
#     nclass = sampler.nclass
#     print("nclass: %d\tnfea:%d" % (nclass, nfeat))

#     # The model
#     model = GCNModel(nfeat=nfeat,
#                      nhid=args.hidden,
#                      nclass=nclass,
#                      nhidlayer=args.nhiddenlayer,
#                      dropout=args.dropout,
#                      baseblock=args.type,
#                      inputlayer=args.inputlayer,
#                      outputlayer=args.outputlayer,
#                      nbaselayer=args.nbaseblocklayer,
#                      activation=F.relu,
#                      withbn=args.withbn,
#                      withloop=args.withloop,
#                      aggrmethod=args.aggrmethod,
#                      mixmode=args.mixmode)

#     optimizer = optim.Adam(model.parameters(),
#                            lr=args.lr, weight_decay=args.weight_decay)

#     # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.618)
#     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)
#     # convert to cuda
#     if args.cuda:
#         model.cuda()

#     # For the mix mode, lables and indexes are in cuda.
#     if args.cuda or args.mixmode:
#         labels = labels.cuda()
#         idx_train = idx_train.cuda()
#         idx_val = idx_val.cuda()
#         idx_test = idx_test.cuda()

#     if args.warm_start is not None and args.warm_start != "":
#         early_stopping = EarlyStopping(fname=args.warm_start, verbose=False)
#         print("Restore checkpoint from %s" % (early_stopping.fname))
#         model.load_state_dict(early_stopping.load_checkpoint())

#     # set early_stopping
#     if args.early_stopping > 0:
#         early_stopping = EarlyStopping(patience=args.early_stopping, verbose=False)
#         print("Model is saving to: %s" % (early_stopping.fname))

#     if args.no_tensorboard is False:
#         tb_writer = SummaryWriter(
#             comment=f"-dataset_{args.dataset}-type_{args.type}"
#         )
#     args.model, args.optimizer, args.sampler, args.labels = model, optimizer, sampler, labels
#     args.idx_train, args.idx_val, args.idx_test, args.early_stopping_class, args.scheduler = idx_train, idx_val, idx_test, early_stopping, scheduler
#     args.tb_writer = tb_writer
#     return args


# def prepare_dataset_onehot_y(dataset):

#     y_set = set()
#     for data in dataset:
#         y_set.add(int(data.y))
#     num_classes = len(y_set)

#     for data in dataset:
#         data.y = F.one_hot(data.y, num_classes=num_classes).to(torch.float)[0]
#     return dataset

# def prepare_dataset_x(dataset):
#     if dataset[0].x is None:
#         max_degree = 0
#         degs = []
#         for data in dataset:
#             degs += [degree(data.edge_index[0], dtype=torch.long)]
#             max_degree = max( max_degree, degs[-1].max().item() )
#             data.num_nodes = int( torch.max(data.edge_index) ) + 1

#         if max_degree < 2000:
#             # dataset.transform = T.OneHotDegree(max_degree)

#             for data in dataset:
#                 degs = degree(data.edge_index[0], dtype=torch.long)
#                 data.x = F.one_hot(degs, num_classes=max_degree+1).to(torch.float)
#         else:
#             deg = torch.cat(degs, dim=0).to(torch.float)
#             mean, std = deg.mean().item(), deg.std().item()
#             for data in dataset:
#                 degs = degree(data.edge_index[0], dtype=torch.long)
#                 data.x = ( (degs - mean) / std ).view( -1, 1 )
#     return dataset

# def GC_model_init(args):

#     data_path = args.GC_data_path
#     dataset_name = args.GC_dataset
#     seed = args.GC_seed
#     lam_range = eval(args.GC_lam_range)
#     log_screen = eval(args.GC_log_screen)
#     gmixup = eval(args.GC_gmixup)
#     num_epochs = args.GC_epoch

#     num_hidden = args.GC_num_hidden
#     batch_size = args.GC_batch_size
#     learning_rate = args.GC_lr
#     ge = args.GC_ge
#     aug_ratio = args.GC_aug_ratio
#     aug_num = args.GC_aug_num
#     model = args.GC_model

#     # if log_screen is True:
#     #     ch = logging.StreamHandler()
#     #     ch.setLevel(logging.DEBUG)
#     #     ch.setFormatter(formatter)
#     #     logger.addHandler(ch)


#     # logger.info('parser.prog: {}'.format(parser.GC_prog))
#     # logger.info("args:{}".format(args))

#     torch.manual_seed(seed)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # logger.info(f"runing device: {device}")

#     path = osp.join(data_path, dataset_name)
#     dataset = TUDataset(path, name=dataset_name)
#     dataset = list(dataset)

#     for graph in dataset:
#         graph.y = graph.y.view(-1)

#     dataset = prepare_dataset_onehot_y(dataset)


#     random.seed(seed)
#     random.shuffle( dataset )

#     train_nums = int(len(dataset) * 0.7)
#     train_val_nums = int(len(dataset) * 0.8)
    
#     avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(dataset[: train_nums])
#     # logger.info(f"avg num nodes of training graphs: { avg_num_nodes }")
#     # logger.info(f"avg num edges of training graphs: { avg_num_edges }")
#     # logger.info(f"avg density of training graphs: { avg_density }")
#     # logger.info(f"median num nodes of training graphs: { median_num_nodes }")
#     # logger.info(f"median num edges of training graphs: { median_num_edges }")
#     # logger.info(f"median density of training graphs: { median_density }")


#     dataset = prepare_dataset_x( dataset )
#     args.org_dataset = dataset

#     # logger.info(f"num_features: {dataset[0].x.shape}" )
#     # logger.info(f"num_classes: {dataset[0].y.shape}"  )

#     num_features = dataset[0].x.shape[1]
#     num_classes = dataset[0].y.shape[0]

#     train_dataset = dataset[:train_nums]
#     random.shuffle(train_dataset)
#     val_dataset = dataset[train_nums:train_val_nums]
#     test_dataset = dataset[train_val_nums:]

#     # logger.info(f"train_dataset size: {len(train_dataset)}")
#     # logger.info(f"val_dataset size: {len(val_dataset)}")
#     # logger.info(f"test_dataset size: {len(test_dataset)}" )


#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)


#     #if model == "GIN":
#     model = GIN(num_features=num_features, num_classes=num_classes, num_hidden=num_hidden).to(device)
#     #else:
#     #    logger.info(f"No model."  )


#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
#     scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

#     args.model, args.optimizer, args.dataset, args.scheduler = model, optimizer, dataset, scheduler
#     args.train_loader, args.val_loader, args.test_loader = train_loader, val_loader, test_loader
#     return args