import pickle as pkl
import sys

import os
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
sys.path.append("../DropEdge/src")
from normalization import fetch_normalization, row_normalize

datadir = "data"

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def obtain_adjacent_matrix_and_degree(graph, normalization):
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # degree = np.asarray(G.degree)
    degree = np.sum(adj, axis=1)
    adj = preprocess_citation(adj, normalization)
    return adj, degree

def load_citation(dataset_str="cora", porting_to_torch=True, data_path=datadir, task_type="full"):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str.lower(), names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))


    x, y, tx, ty, allx, ally = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    if task_type == "full":
        print("Load full supervised task.")
        #supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(ally)- 500)
        idx_val = range(len(ally) - 500, len(ally))
    elif task_type == "semi":
        print("Load semi-supervised task.")
        #semi-supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)
    else:
        raise ValueError("Task type: %s is not supported. Available option: full and semi.")

    features = row_normalize(features)
    features = np.array(features.todense())
    labels = np.argmax(labels, axis=1)
    # porting to pytorch
    if porting_to_torch:
        features = torch.FloatTensor(features).float()
        labels = torch.LongTensor(labels)
        # labels = torch.max(labels, dim=1)[1]
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

    learning_type = "transductive"
    return features, labels, idx_train, idx_val, idx_test, learning_type

def merge_graphs(graph, hypergraph):

    for node, neighbors in hypergraph.items():
        if node not in graph:
            graph[node] = neighbors
        else:
            graph[node] += neighbors
            graph[node] = list(set(graph[node]))
    return graph

def load_citation_adj(data_path=datadir, normalization="AugNormAdj", hypergraph=None, dataset_str="cora", porting_to_torch=True):

    with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str.lower(), 'graph')), 'rb') as f:
        if sys.version_info > (3, 0):
            graph = pkl.load(f, encoding='latin1')
        else:
            graph = pkl.load(f)
    if hypergraph is not None:
        graph = merge_graphs(graph, hypergraph)
    adj, degree = obtain_adjacent_matrix_and_degree(graph, normalization)
    if porting_to_torch:
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        degree = torch.LongTensor(degree)

    return adj, degree

def sgc_precompute(features, adj, degree):
    #t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = 0 #perf_counter()-t
    return features, precompute_time

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)


def loadRedditFromNPZ(dataset_dir=datadir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir +"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']


def load_reddit_data(normalization="AugNormAdj", porting_to_torch=True, data_path=datadir):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ(data_path)
    labels = np.zeros(adj.shape[0])
    labels[train_index]  = y_train
    labels[val_index]  = y_val
    labels[test_index]  = y_test
    adj = adj + adj.T + sp.eye(adj.shape[0])
    train_adj = adj[train_index, :][:, train_index]
    degree = np.sum(train_adj, axis=1)

    features = torch.FloatTensor(np.array(features))
    features = (features-features.mean(dim=0))/features.std(dim=0)
    train_features = torch.index_select(features, 0, torch.LongTensor(train_index))
    if not porting_to_torch:
        features = features.numpy()
        train_features = train_features.numpy()

    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    train_adj = adj_normalizer(train_adj)

    if porting_to_torch:
        train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
        labels = torch.LongTensor(labels)
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        degree = torch.LongTensor(degree)
        train_index = torch.LongTensor(train_index)
        val_index = torch.LongTensor(val_index)
        test_index = torch.LongTensor(test_index)
    learning_type = "inductive"
    return adj, train_adj, features, train_features, labels, train_index, val_index, test_index, degree, learning_type


    
def data_loader(dataset, data_path=datadir, normalization="AugNormAdj", porting_to_torch=True, task_type = "full"):
    if dataset == "reddit":
        return load_reddit_data(normalization, porting_to_torch, data_path)
    else:
        (features,
         labels,
         idx_train,
         idx_val,
         idx_test,
         learning_type) = load_citation(dataset, porting_to_torch, data_path, task_type)

        adj, degree = load_citation_adj(data_path, "AugNormAdj", None, dataset, False)
        train_adj = adj
        train_features = features
        return adj, train_adj, features, train_features, labels, idx_train, idx_val, idx_test, degree, learning_type


def update_adj_and_degree(hypergraph, dataset, data_path=datadir):
    adj, degree = load_citation_adj(data_path, "AugNormAdj", hypergraph, dataset, False)
    train_adj = adj
    return adj, train_adj, degree

