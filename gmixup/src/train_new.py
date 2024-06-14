from time import time
import logging
import os
import os.path as osp
import numpy as np
import time

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset,Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from torch.autograd import Variable

import random
from torch.optim.lr_scheduler import StepLR

import sys
sys.path.append('../gmixup')
from gmixup.src.utils import stat_graph, split_class_graphs, align_graphs
from gmixup.src.utils import two_graphons_mixup, universal_svd
from gmixup.src.graphon_estimator import universal_svd
from gmixup.src.models import GIN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d')



def prepare_dataset_x(dataset):
    if dataset[0].x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max( max_degree, degs[-1].max().item() )
            data.num_nodes = int( torch.max(data.edge_index) ) + 1

        if max_degree < 2000:
            # dataset.transform = T.OneHotDegree(max_degree)

            for data in dataset:
                degs = degree(data.edge_index[0], dtype=torch.long)
                data.x = F.one_hot(degs, num_classes=max_degree+1).to(torch.float)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            for data in dataset:
                degs = degree(data.edge_index[0], dtype=torch.long)
                data.x = ( (degs - mean) / std ).view( -1, 1 )
    return dataset



def prepare_dataset_onehot_y(dataset):

    y_set = set()
    for data in dataset:
        y_set.add(int(data.y))
    num_classes = len(y_set)

    for data in dataset:
        data.y = F.one_hot(data.y, num_classes=num_classes).to(torch.float)[0]
    return dataset


def mixup_cross_entropy_loss(input, target, size_average=True):
    """Origin: https://github.com/moskomule/mixup.pytorch
    in PyTorch's cross entropy, targets are expected to be labels
    so to predict probabilities this loss is needed
    suppose q is the target and p is the input
    loss(p, q) = -\sum_i q_i \log p_i
    """
    assert input.size() == target.size()
    assert isinstance(input, Variable) and isinstance(target, Variable)
    loss = - torch.sum(input * target)
    return loss / input.size()[0] if size_average else loss




def train(model, train_loader,optimizer,num_classes):
    model.train()
    loss_all = 0
    graph_all = 0
    for data in train_loader:
        # print( "data.y", data.y )
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        y = data.y.view(-1, num_classes)
        loss = mixup_cross_entropy_loss(output, y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        graph_all += data.num_graphs
        optimizer.step()
    loss = loss_all / graph_all
    return model, loss


def test(model, loader,num_classes):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        y = data.y.view(-1, num_classes)
        loss += mixup_cross_entropy_loss(output, y).item() * data.num_graphs
        y = y.max(dim=1)[1]
        correct += pred.eq(y).sum().item()
        total += data.num_graphs
    acc = correct / total
    loss = loss / total
    return acc, loss


def training_and_test(args):
    learning_rate = args.GC_lr

    seed = args.GC_seed
    num_epochs = args.GC_epoch

    model = args.model

    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = args.dataset
    # dataset = list(dataset)

    # for graph in dataset:
    #     graph.y = graph.y.view(-1)

    # dataset = prepare_dataset_onehot_y(dataset)


    random.seed(seed)
    random.shuffle( dataset )

    train_nums = int(len(dataset) * 0.7)
    train_val_nums = int(len(dataset) * 0.8)
    
    avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(dataset[: train_nums])
    logger.info(f"avg num nodes of training graphs: { avg_num_nodes }")
    logger.info(f"avg num edges of training graphs: { avg_num_edges }")
    logger.info(f"avg density of training graphs: { avg_density }")
    logger.info(f"median num nodes of training graphs: { median_num_nodes }")
    logger.info(f"median num edges of training graphs: { median_num_edges }")
    logger.info(f"median density of training graphs: { median_density }")

    # resolution = int(median_num_nodes)

    # dataset = prepare_dataset_x( dataset )

    logger.info(f"num_features: {dataset[0].x.shape}" )
    logger.info(f"num_classes: {dataset[0].y.shape}"  )

    num_features = dataset[0].x.shape[1]
    num_classes = dataset[0].y.shape[0]

    train_dataset = dataset[:train_nums]
    random.shuffle(train_dataset)
    val_dataset = dataset[train_nums:train_val_nums]
    test_dataset = dataset[train_val_nums:]

    logger.info(f"train_dataset size: {len(train_dataset)}")
    logger.info(f"val_dataset size: {len(val_dataset)}")
    logger.info(f"test_dataset size: {len(test_dataset)}" )


    train_loader = DataLoader(train_dataset, batch_size=args.GC_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.GC_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.GC_batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

    for epoch in range(1, num_epochs):
        model, train_loss = train(model, train_loader,optimizer=optimizer,num_classes=num_classes)
        train_acc = 0
        val_acc, val_loss = test(model, val_loader,num_classes=num_classes)
        test_acc, test_loss = test(model, test_loader,num_classes=num_classes)
        scheduler.step()

    return train_loss, val_loss, test_loss, val_acc, test_acc, model
