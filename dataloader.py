# -*- coding: utf-8 -*-
"""
Dataloader of CPF datasets are adapted from the CPF implementation
https://github.com/BUPT-GAMMA/CPF/tree/389c01aaf238689ee7b1e5aba127842341e123b6/data

Dataloader of NonHom datasets are adapted from the Non-homophily benchmarks
https://github.com/CUAI/Non-Homophily-Benchmarks

Dataloader of BGNN datasets are adapted from the BGNN implementation and dgl example of BGNN
https://github.com/nd7141/bgnn
https://github.com/dmlc/dgl/tree/473d5e0a4c4e4735f1c9dc9d783e0374328cca9a/examples/pytorch/bgnn
"""

# +
import numpy as np
import scipy.sparse as sp
import torch
import dgl
import os
import scipy
import pandas as pd
import json
from dgl.data.utils import load_graphs
from os import path
from category_encoders import CatBoostEncoder
from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing
from data_preprocess import (
    normalize_adj,
    eliminate_self_loops_adj,
    largest_connected_components,
    binarize_labels,
)
from ogb.nodeproppred import DglNodePropPredDataset


# nifty imports
import os
import os.path as osp
from os import environ
import time

import torch
from sklearn.metrics import roc_auc_score

import torch_geometric.transforms as T
from torch_geometric.datasets import CitationFull, Coauthor, WebKB, LastFMAsia, Twitch
# from dblp_fairness import DBLPFairness
from say_no import SayNo
from nifty import Nifty

# from gcn_conv import GCNConv
from torch_geometric.utils import negative_sampling, to_dense_adj, add_remaining_self_loops, is_undirected, dense_to_sparse, subgraph, contains_self_loops, coalesce
from torch_geometric.data import Data
from connected_classes import ConnectedClasses, LargestConnectedComponents, LargestBiconnectedComponents

import seaborn as sns
import matplotlib.pyplot as plt
# import plotly.graph_objs as go
# import plotly.express as px
import math
import numpy as np
import networkx as nx
import re
import pickle as pkl

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

# my imports
import torch_geometric
from torch_geometric.utils.convert import to_dgl
import random

# sbm import
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import StochasticBlockModelDataset
from torch_geometric.utils import stochastic_blockmodel_graph
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.utils import to_networkx
import shutil
# -


CPF_data = ["cora", "citeseer", "pubmed", "a-computer", "a-photo"]
OGB_data = ["ogbn-arxiv", "ogbn-products"]
NonHom_data = ["pokec", "penn94"]
BGNN_data = ["house_class", "vk_class"]
Nifty_data = ["German", "Credit"]
Say_no_data = ["Pokec-z", "Pokec-n", "NBA"]
Synthetic_data = ["syn-1", "syn-2"]
Twitter_data = ["sport", "occupation"]
SBM = ["sbm"]


def load_data(dataset, dataset_path, **kwargs):
    if dataset in CPF_data:
        return load_cpf_data(
            dataset,
            dataset_path,
            kwargs["seed"],
            kwargs["labelrate_train"],
            kwargs["labelrate_val"],
            kwargs["p"],
            kwargs["q"],
            kwargs["c"],
        )
    elif dataset in OGB_data:
        return load_ogb_data(dataset, dataset_path)
    elif dataset in NonHom_data:
        return load_nonhom_data(dataset, dataset_path, kwargs["split_idx"])
    elif dataset in BGNN_data:
        return load_bgnn_data(dataset, dataset_path, kwargs["split_idx"])
    elif dataset in Nifty_data:
        return load_nifty_data(dataset)
    elif dataset in Say_no_data:
        return load_say_no_data(dataset)
    elif dataset in Synthetic_data:
        return load_synthetic_data(dataset)
    elif dataset in Twitter_data:
        return load_twitter(dataset)
    elif dataset in SBM:
        return load_sbm(dataset, kwargs["p"], kwargs["q"], kwargs["c"])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_ogb_data(dataset, dataset_path):
    data = DglNodePropPredDataset(dataset, dataset_path)
    splitted_idx = data.get_idx_split()
    idx_train, idx_val, idx_test = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )

    g, labels = data[0]
    labels = labels.squeeze()

    # Turn the graph to undirected
    if dataset == "ogbn-arxiv":
        srcs, dsts = g.all_edges()
        g.add_edges(dsts, srcs)
        g = g.remove_self_loop().add_self_loop()

    return g, labels, idx_train, idx_val, idx_test


# +
def _load_data(dataset, sens_attr, predict_attr, sens_number, label_number, preprocess_func=None, test_idx=False):
    name = dataset
    path = f'./data/{name}'

    print('Loading {} dataset from {}'.format(name, path))

    features, labels, idx_features_labels = get_features_and_labels(dataset, sens_attr, predict_attr, preprocess_func)

    idx_train, idx_val, idx_test, idx_map, sens, idx_sens_train = split_data(dataset, idx_features_labels, sens_number, labels, label_number)

    adj = build_adjacency(dataset, labels.shape[0], idx_map)

    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train

def get_features_and_labels(dataset, sens_attr, predict_attr, preprocess_func=None):
    name = dataset
    path = f'./data/{name}'
    
    idx_features_labels = pd.read_csv(os.path.join(path, f"{name}.csv"))
    header = list(idx_features_labels.columns)
    if preprocess_func:
        idx_features_labels, header = preprocess_func(idx_features_labels, header, predict_attr, sens_attr)
    # print(idx_features_labels[header], type(idx_features_labels[header]))
    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    # sens_features = idx_features_labels[sens_attr]

    labels = idx_features_labels[predict_attr]
    labels[labels == -1] = 0
    labels[labels > 1] = 1

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

#         # feature normalization for some datasets
#         if self.name == 'nba':
#             features = feature_norm(features)
#         elif self.name in ['bail','credit']:        
#             norm_features = feature_norm(features)
#             norm_features[:, self.config['sens_idx']] = features[:, self.config['sens_idx']]
#             features = norm_features

    return features, labels, idx_features_labels

def split_data(dataset, idx_features_labels, sens_number, labels, label_number, test_idx=False):
    name = dataset

    idx = np.arange(labels.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}

    # split setting for semi-synthetic datasets in NIFTYGNN
    if name in ['german', 'credit', 'bail', 'germanA', 'creditA', 'bailA']:
        label_idx_0 = np.where(labels==0)[0]
        label_idx_1 = np.where(labels==1)[0]
        random.shuffle(label_idx_0)
        random.shuffle(label_idx_1)
        idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
        idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])
        sens = idx_features_labels[self.config['sens_attr']].values.astype(int)
        sens_idx = set(np.where(sens >= 0)[0])
    # split setting for the real-world datasets in FairGNN
    elif name in ['pokecz_z', 'pokec_n', 'nba']:
        idx = np.array(idx_features_labels["user_id"], dtype=int)   # print(idx)
        idx_map = {j: i for i, j in enumerate(idx)}        
        label_idx = np.where(labels>=0)[0]
        random.shuffle(label_idx)
        idx_train = label_idx[:min(int(0.5 * len(label_idx)),label_number)]
        idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
        if test_idx:
            idx_test = label_idx[label_number:]
            idx_val = idx_test
        else:
            idx_test = label_idx[int(0.75 * len(label_idx)):]
        sens = idx_features_labels[self.config['sens_attr']].values
        sens_idx = set(np.where(sens >= 0)[0])
        idx_test = np.asarray(list(sens_idx & set(idx_test)))
    # split setting in our benchmark
    else:
        print("here!")
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}
        
        # need to convert labels to numpy
        labels_new = labels.tolist()
        labels_new = np.array(labels_new)
        
        label_idx = np.where(labels_new>=0)[0]   
        random.shuffle(label_idx)
        n = len(label_idx)
        train_ratio = 0.6 # for occupation, sport
        idx_train = label_idx[:int(n*train_ratio)]
        idx_val = label_idx[int(n*train_ratio): int(n*(1+train_ratio)/2)]
        idx_test = label_idx[int(n*(1+train_ratio)/2):]
        if dataset == "sport":
            sens_attr = "race"
        else: # occupation
            sens_attr = "gender"
        sens = idx_features_labels[sens_attr].values    #print(sens, type(sens))
        sens_idx = set(np.where(sens >= 0)[0])
        #idx_test = np.asarray(list(sens_idx & set(idx_test)))

    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.shuffle(idx_sens_train)    
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return idx_train, idx_val, idx_test, idx_map, sens, idx_sens_train

def build_adjacency(dataset, num_nodes, idx_map):

    name = dataset
    path = f'./data/{name}'

    if name in ['sport', 'occupation']:
        edges_unordered = np.genfromtxt(os.path.join(path, f'{name}_edges.txt'), dtype=int)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(num_nodes, num_nodes),
                            dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])
    # load adj from adjacency matrix (.npz files)
    elif name in ['germanA', 'creditA', 'bailA']:
        adj = load_npz(f'{path}/{name}_edges.npz')
    else:
        raise ValueError(f"No adjacency build function for this dataset: {name}.")

    return adj


# -

def load_twitter(dataset):
    def _preprocess_twitter(idx_features_labels, header, predict_attr, sens_attr=None):
        header.remove(predict_attr)
        header.remove('user_id')
        header.remove('embeddings')
        return idx_features_labels, header
        
    if dataset == "sport":
        sens_attr = "race"
        pred_attr = "sport"
        sens_num = 3508
        label_num = 3508
    else: # occupation
        sens_attr = "gender"
        pred_attr = "area"
        sens_num = 6951
        label_num = 6951
        
    
    adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = _load_data(dataset, sens_attr, pred_attr, 
                           sens_num, label_num, 
                           preprocess_func=_preprocess_twitter)
    
    g = dgl.from_scipy(adj)
    g.ndata['feat'] = features

    return g, labels, idx_train, idx_val, idx_test, sens

# +
from torch_geometric.utils import to_undirected
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import StochasticBlockModelDataset
from torch_geometric.utils import stochastic_blockmodel_graph

class CustomSBMDataset(StochasticBlockModelDataset):
    
    def __init__(self, *args, c=None, p=0.3, q=0.50, num_nodes=3000, **kwargs):
        self.c = c # class balance ratio
        self.p = p  # group balance ratio
        self.q = q  # edge probability ratio
        self.num_nodes = num_nodes        
        super().__init__(*args, **kwargs)
        
    @property
    def raw_file_names(self):
        return ['dummy_file']

    @property
    def processed_file_names(self):
        return [f'data_p={self.p}_q={self.q}c={self.c}.pt']
    
    def process(self):
        print("SBM dataset")
        # Define class (y) and group (a) values
        a_s = [0, 1]
        y_s = [0, 1]
        
        # Set up weights for classes and groups
        class_weights = torch.tensor([self.c, 1-self.c])
        print("class weights", class_weights)
        group_weights = torch.tensor([self.p, 1-self.p])  # p for group weights
        print("group weights", group_weights)
        
        # Calculate block sizes for each combination of class and group
        # Format: [class0_group0, class0_group1, class1_group0, class1_group1]
        block_sizes = torch.tensor([
            int(self.num_nodes * class_weights[0] * group_weights[0]),  # y=0, a=0
            int(self.num_nodes * class_weights[0] * group_weights[1]),  # y=0, a=1
            int(self.num_nodes * class_weights[1] * group_weights[0]),  # y=1, a=0
            int(self.num_nodes * class_weights[1] * group_weights[1])   # y=1, a=1
        ], dtype=torch.long)
        
        # Make sure the block sizes sum to num_nodes
        if block_sizes.sum() != self.num_nodes:
            diff = self.num_nodes - block_sizes.sum()
            block_sizes[-1] += diff  # Adjust the last block to make it sum to num_nodes
            
        # Define edge probabilities
        intra_group = 0.1  # Base probability for connections within same group
        inter_group = float(intra_group * self.q)  # Reduced probability for between groups
        # q values: 0.05, 0.25, 0.50, 0.75, 1.00
        
        # Create edge probability matrix (4x4) for all combinations of blocks
        # Order: [y0a0, y0a1, y1a0, y1a1] x [y0a0, y0a1, y1a0, y1a1]
        # The edge probabilities follow the pattern in the paper
        edge_probs = torch.zeros((4, 4))
        
        # Fill the edge probability matrix according to the paper's description
        for i in range(4):
            for j in range(4):
                # Extract the group (a) for each block
                a_i = i % 2
                a_j = j % 2
                
                # Set edge probability based on whether the groups match
                if a_i == a_j:
                    edge_probs[i, j] = intra_group
                else:
                    edge_probs[i, j] = inter_group
        
        print("intra_group",intra_group)
        print("inter_group",inter_group)
        
        print("edge probs", edge_probs)
        print("block sizes", block_sizes)
        
        # Generate the SBM graph
        edge_index = stochastic_blockmodel_graph(
            block_sizes=block_sizes,
            edge_probs=edge_probs,
            directed=False
        )
        
        # Initialize labels
        group_labels = torch.zeros(self.num_nodes, dtype=torch.long)
        class_labels = torch.zeros(self.num_nodes, dtype=torch.long)

        # Assign class and group labels based on block structure
        start_idx = 0
        for i, block_size in enumerate(block_sizes):
            end_idx = start_idx + block_size
            class_id = i // 2  # Integer division to get class ID (0 or 1)
            group_id = i % 2   # Modulo to get group ID (0 or 1)
            
            class_labels[start_idx:end_idx] = class_id
            group_labels[start_idx:end_idx] = group_id
            
            start_idx = end_idx

        # Generate features
        sig_core = 10  # Covariance, constant for preliminary experimentation, can vary it later
        sig_spu = 1
        d = 100  # Number of core features
        num_channels = 2 * d

        # Initialize features with random values
        x = torch.randn(self.num_nodes, num_channels)

        # Adjust features based on class and group following the paper's approach
        for y in y_s:  # y is the class label
            for a in a_s:  # a is the group label
                # Modify core features based on class (y)
                mask = (class_labels == y) & (group_labels == a)
                if y == 0:
                    x[mask, :d] = sig_core * x[mask, :d] - 1 # y=0 data centered at -1
                else:
                    x[mask, :d] = sig_core * x[mask, :d] + 1 # y=1 data centered at +1

                # Modify spurious features based on group (a)
                if a == 0:
                    x[mask, d:] = sig_spu * x[mask, d:] - 1 # a=0 data centered at -1
                else:
                    x[mask, d:] = sig_spu * x[mask, d:] + 1 # a=1 data centered at +1
                    
        from sklearn.decomposition import PCA
        x_ = x.tolist()
        x_ = np.array(x_)
        class_labels_ = class_labels.tolist()
        class_labels_ = np.array(class_labels_)
        
        pca = PCA()
        Xt = pca.fit_transform(x_)
        plot = plt.scatter(Xt[:,0], Xt[:,1], c=class_labels_)
        plt.savefig(f'data_distribution.png')
        plt.close()

        # Create the final data object
        data = Data(x=x, edge_index=edge_index, y=class_labels, group=group_labels)

        # Save the data
        data_list = [data]
        torch.save(self.collate(data_list), self.processed_paths[0])


# -

def check_sbm_edge_probabilities(edge_index, class_labels, group_labels,q_val):
    """
    Helper function to verify edge probabilities in a graph using PyTorch only
    
    Parameters:
    -----------
    edge_index : torch.Tensor
        Edge index tensor of shape [2, num_edges]
    class_labels : torch.Tensor or list-like
        Class labels for each node (y)
    group_labels : torch.Tensor or list-like
        Group labels for each node (a)
    """
    import torch
    
    # Convert inputs to PyTorch tensors with explicit dtype
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    if not isinstance(class_labels, torch.Tensor):
        class_labels = torch.tensor(class_labels, dtype=torch.long)
    
    if not isinstance(group_labels, torch.Tensor):
        group_labels = torch.tensor(group_labels, dtype=torch.long)
    
    # Count nodes in each block
    block_counts = torch.zeros(2, 2, dtype=torch.float)  # [class, group]
    for y in [0, 1]:
        for a in [0, 1]:
            block_counts[y, a] = float(((class_labels == y) & (group_labels == a)).sum().item())
            # print(f"Block (y={y}, a={a}) has {int(block_counts[y, a])} nodes")
    
    # Get source and target nodes
    src_nodes, tgt_nodes = edge_index
    
    # Initialize edge count tensor
    edge_counts = torch.zeros(2, 2, 2, 2, dtype=torch.float)  # [y_src, a_src, y_tgt, a_tgt]
        
    # Count edges between each block pair
    for i in range(edge_index.shape[1]):
        src = src_nodes[i].item()
        tgt = tgt_nodes[i].item()
            
        y_src = class_labels[src].item()
        y_tgt = class_labels[tgt].item()
        a_src = group_labels[src].item()
        a_tgt = group_labels[tgt].item()
            
        edge_counts[y_src, a_src, y_tgt, a_tgt] += 1
        
    # Calculate and print observed edge probabilities
    # print("\n--- Checking Edge Probabilities ---")
    all_within_threshold = True
    failed_checks = []

    for y_src in [0, 1]:
        for y_tgt in [0, 1]:
            # print(f"\n(y_src == {y_src}) & (y_tgt == {y_tgt})")
            for a_src in [0, 1]:
                for a_tgt in [0, 1]:
                    count = edge_counts[y_src, a_src, y_tgt, a_tgt].item()

                    max_edges = block_counts[y_src, a_src] * block_counts[y_tgt, a_tgt]
                    # Calculate probability
                    prob = count / max_edges
                    # Calculate the expected edge probability
                    expected = 0.1  # intra
                    if a_src != a_tgt:
                        expected *= q_val  # inter
                    # Statistical calculations
                    std_error = math.sqrt(2*(expected * (1-expected)) / max_edges)
                    threshold = 3 * std_error  # ~98% confidence interval

                    # Check if within threshold
                    within_threshold = abs(expected - prob) <= threshold
                    if not within_threshold:
                        all_within_threshold = False
                        failed_checks.append((y_src, a_src, y_tgt, a_tgt, prob, expected, abs(expected-prob), threshold))

                    print(f"P(edge | y_src={y_src}, a_src={a_src}, y_tgt={y_tgt}, a_tgt={a_tgt}): {prob:.6f}, should be {expected:.6f}, diff:{abs(expected-prob):.6f}{' ✓' if within_threshold else ' ✗'}")

    # Final assertion
#     assert all_within_threshold, f"Some edge probabilities were outside the expected threshold: {failed_checks}"
    assert len(failed_checks) < 2, f"Some edge probabilities were outside the expected threshold: {failed_checks}"


def load_sbm(dataset, p, q, c):
    print("q (edge prob ratio):",q)
    print("p (group balance):",p)
    
    num_channels = 200
    num_nodes = 3000
    
    # Set up dataset name and path
    dataset_name = f"sbm_p{p}_q{q}_class{c}_nodes{num_nodes}_channels{num_channels}"
    dataset_path = os.path.join('./data/sbm', dataset_name)

    # Create the dataset
    dataset = CustomSBMDataset(
        root=dataset_path,
        block_sizes=[],
        p=p,
        q=q,
        num_nodes=num_nodes,
        edge_probs=[],
        num_channels=num_channels,
        num_graphs=1,
        is_undirected=True,
        transform=None,
        pre_transform=None,
        c=c
    )

    # Access the generated graph
    data = dataset[0]
    
    # Convert to DGL graph if needed
    g = dgl.from_networkx(to_networkx(data))
    g.ndata['feat'] = data.x
    
    # Prepare indices
    num_nodes = data.num_nodes
    idx_train = torch.randperm(num_nodes)[:int(0.6 * num_nodes)]
    idx_val = torch.randperm(num_nodes)[int(0.6 * num_nodes):int(0.8 * num_nodes)]
    idx_test = torch.randperm(num_nodes)[int(0.8 * num_nodes):]
    
    return (g, 
            data.y, 
            idx_train, 
            idx_val, 
            idx_test, 
            data.group)


def load_synthetic_data(dataset, train_ratio=0.6): # future experiments: adjust train ratio
    name = dataset
    path = f'./data/{name}'
    
    print('Loading {} dataset from {}'.format(name, path))
    
    # load from saved data
    labels = np.loadtxt(f'{path}/{name}_label.txt', dtype=int)
    sens = np.loadtxt(f'{path}/{name}_sens.txt', dtype=int)
    features = np.loadtxt(f'{path}/{name}_feat.csv', delimiter=',')
    edges = np.loadtxt(f'{path}/{name}_edges.txt', delimiter=',', dtype=int)
    adj_coo = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(features.shape[0], features.shape[0]))
    adj = adj_coo.tocsr()
    
    n = features.shape[0]
    adj = adj + sp.eye(n)
    features = sp.csr_matrix(features, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    idx = np.arange(n)
    np.random.shuffle(idx)
    idx_train = idx[:int(n*train_ratio)]
    idx_val = idx[int(n*train_ratio): int(n*(1+train_ratio)/2)]
    idx_test = idx[int(n*(1+train_ratio)/2):]
    sens_idx = set(np.where(sens >= 0)[0])
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    g = dgl.from_scipy(adj)
    g.ndata['feat'] = features
    
    return g, labels, idx_train, idx_val, idx_test, sens


# old experiment: quantiles/degree experimentation
def load_cpf_data(dataset, dataset_path, seed, labelrate_train, labelrate_val):
    data_path = Path.cwd().joinpath(dataset_path, f"{dataset}.npz")
    if os.path.isfile(data_path):
        data = load_npz_to_sparse_graph(data_path)
    else:
        raise ValueError(f"{data_path} doesn't exist.")

    data = data.standardize()
    adj, features, labels = data.unpack()

    print("labels before: ", labels)
    print("classes: ", np.unique(labels))
    print("labels shape: ", labels.shape)

    labels = label_binarize(labels, classes=np.unique(labels))
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.label_binarize.html
    
    print("labels after: ", labels)
    print("labels shape: ", labels.shape)
    
    print("dataloader labels: ", labels)

    random_state = np.random.RandomState(seed)
    idx_train, idx_val, idx_test = get_train_val_test_split(
        random_state, labels, labelrate_train, labelrate_val
    )

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels.argmax(axis=1))
    
    adj = normalize_adj(adj)
    adj_sp = adj.tocoo()

    row = torch.LongTensor(adj_sp.row)
    col = torch.LongTensor(adj_sp.col)
    
    g = dgl.graph((row, col))
    g.ndata["feat"] = features

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
        
    # Calculate degree of each node
    degrees = g.in_degrees()  # or g.out_degrees() for out-degrees

    # Add degree as a node attribute
    g.ndata["degree"] = degrees.float()  # Assuming you want to store degrees as floats
    
    sens = g.ndata["degree"]
    
    print("sens, ", sens)
    print("median: ", torch.median(sens))

    # lower k% quantile (torch)
    sorted_data = torch.sort(sens)[0] # change this
    quantile_int = int(quantile)
    quantile_percentile = quantile_int / 100
    k = int(len(sorted_data) * quantile_percentile)
    print("in dataloader, quantile percentile: ", quantile_percentile)
    sens_lower_quantile = sorted_data[k]
    res_tensor = torch.zeros_like(sens)
    res_tensor[sens > sens_lower_quantile] = 1
    
    print("res tens: ", res_tensor)
    
    print("end of dataloader")
    
    sens = res_tensor
    
    return g, labels, idx_train, idx_val, idx_test, sens # added sens


def load_nonhom_data(dataset, dataset_path, split_idx):
    data_path = Path.cwd().joinpath(dataset_path, f"{dataset}.mat")
    data_split_path = Path.cwd().joinpath(
        dataset_path, "splits", f"{dataset}-splits.npy"
    )

    if dataset == "pokec":
        g, features, labels = load_pokec_mat(data_path)
    elif dataset == "penn94":
        g, features, labels = load_penn94_mat(data_path)
    else:
        raise ValueError("Invalid dataname")

    g = g.remove_self_loop().add_self_loop()
    g.ndata["feat"] = features
    labels = torch.LongTensor(labels)

    splitted_idx = load_fixed_splits(dataset, data_split_path, split_idx)
    idx_train, idx_val, idx_test = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    return g, labels, idx_train, idx_val, idx_test


def load_bgnn_data(dataset, dataset_path, split_idx):
    data_path = Path.cwd().joinpath(dataset_path, f"{dataset}")

    g, X, y, cat_features, masks = read_input(data_path)
    train_mask, val_mask, test_mask = (
        masks[str(split_idx)]["train"],
        masks[str(split_idx)]["val"],
        masks[str(split_idx)]["test"],
    )

    encoded_X = X.copy()
    if cat_features is not None and len(cat_features):
        encoded_X = encode_cat_features(
            encoded_X, y, cat_features, train_mask, val_mask, test_mask
        )
    encoded_X = normalize_features(encoded_X, train_mask, val_mask, test_mask)
    encoded_X = replace_na(encoded_X, train_mask)
    features, labels = pandas_to_torch(encoded_X, y)

    g = g.remove_self_loop().add_self_loop()
    g.ndata["feat"] = features
    labels = labels.long()

    idx_train = torch.LongTensor(train_mask)
    idx_val = torch.LongTensor(val_mask)
    idx_test = torch.LongTensor(test_mask)
    return g, labels, idx_train, idx_val, idx_test


# +
from typing import List, Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform

class NormalizeFeatures(BaseTransform):
    def __init__(self, attrs: List[str] = ["x"]):
        self.attrs = attrs

    def __call__(self, data: Data) -> Data:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                if value.numel() > 0:
                    min_values = value.min(dim=0)[0]
                    max_values = value.max(dim=0)[0]
                    store[key] = 2 * (value - min_values).div(max_values - min_values) - 1
        return data


# -

def load_say_no_data(dataset_name):
    
    path = osp.join('.', 'data', 'SayNo')
    dataset = SayNo(path, name=dataset_name) #, transform=transform)
    
    dataset = dataset.shuffle()
    
    data = dataset[0]    
    g = to_dgl(data)
    X = data.x
    y = data.y
    
    g.ndata["feat"] = X[:, :-1] # this is everything except the sens attr
    
    labels = y
    sens = data.sens
    
    if dataset_name == "NBA": # these numbers are based on prev work
        label_number = 100
        sens_number = 50
    else:
        label_number = 500
        sens_number = 200

    random.seed(20)
    print("unique 1: ", labels.unique()[0])
    if dataset_name != "NBA":
        labels[labels < 2] = 0
        labels[labels >= 2] = 1
    label_idx = np.where(np.array(labels.tolist())>=0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[:min(int(0.5 * len(label_idx)),label_number)]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
   
    if dataset_name == "NBA": 
        idx_test = label_idx[label_number:]
        idx_val = idx_test
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)):]

    sens_idx = set(np.where(np.array(sens.tolist()) >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens.tolist())
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(20)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])


    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    return g, labels, idx_train, idx_val, idx_test, sens # this is the sensitive attr


def load_nifty_data(dataset_name):
    
    path = osp.join('.', 'data', 'Nifty')
    dataset = Nifty(path, name=dataset_name) #, transform=transform)
    
    dataset = dataset.shuffle()
    
    data = dataset[0]    
    g = to_dgl(data)
    X = data.x
    y = data.y
    
    g.ndata["feat"] = X[:, :-1] # this is everything except the sens attr
    
    labels = y.long()
    sens = X[:, -1]

    random.seed(20)
    label_idx_0 = np.array(torch.where(labels==0)[0].tolist())
    label_idx_1 = np.array(torch.where(labels==1)[0].tolist())
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    label_number = 1000

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])
    
    idx_train = torch.tensor(idx_train.tolist()).long()
    idx_val = torch.tensor(idx_val.tolist()).long()
    idx_test = torch.tensor(idx_test.tolist()).long()
    
    return g, labels, idx_train, idx_val, idx_test, sens # this is the sensitive attr


# +
def load_out_t(out_t_dir):
    load = np.load(out_t_dir.joinpath("out.npz"))
    # np -> list
    my_list = load["arr_0"].tolist()
    # list -> tensor
    my_tensor = torch.tensor(my_list)
    
    return my_tensor

#     return torch.from_numpy(np.load(out_t_dir.joinpath("out.npz"))["arr_0"])


# -

""" For NonHom"""
dataset_drive_url = {"pokec": "1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y"}
splits_drive_url = {"pokec": "1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_"}


def load_penn94_mat(data_path):
    mat = scipy.io.loadmat(data_path)
    A = mat["A"]
    metadata = mat["local_info"]

    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(np.int)

    # make features into one-hot encodings
    feature_vals = np.hstack((np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    g = dgl.graph((edge_index[0], edge_index[1]))
    g = dgl.to_bidirected(g)

    features = torch.tensor(features, dtype=torch.float)
    labels = torch.tensor(metadata[:, 1] - 1)  # gender label, -1 means unlabeled
    return g, features, labels


def load_pokec_mat(data_path):
    if not path.exists(data_path):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url["pokec"], dest_path=data_path, showsize=True
        )

    fulldata = scipy.io.loadmat(data_path)
    edge_index = torch.tensor(fulldata["edge_index"], dtype=torch.long)
    g = dgl.graph((edge_index[0], edge_index[1]))
    g = dgl.to_bidirected(g)

    features = torch.tensor(fulldata["node_feat"]).float()
    labels = fulldata["label"].flatten()
    return g, features, labels


class NCDataset(object):
    def __init__(self, name, root):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type="random", train_prop=0.5, valid_prop=0.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == "random":
            ignore_negative = False if self.name == "ogbn-proteins" else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label,
                train_prop=train_prop,
                valid_prop=valid_prop,
                ignore_negative=ignore_negative,
            )
            split_idx = {"train": train_idx, "valid": valid_idx, "test": test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, len(self))


def load_fixed_splits(dataset, data_split_path="", split_idx=0):
    if not os.path.exists(data_split_path):
        assert dataset in splits_drive_url.keys()
        gdd.download_file_from_google_drive(
            file_id=splits_drive_url[dataset], dest_path=data_split_path, showsize=True
        )

    splits_lst = np.load(data_split_path, allow_pickle=True)
    splits = splits_lst[split_idx]

    for key in splits:
        if not torch.is_tensor(splits[key]):
            splits[key] = torch.as_tensor(splits[key])

    return splits


"""For BGNN """


def pandas_to_torch(*args):
    return [torch.from_numpy(arg.to_numpy(copy=True)).float().squeeze() for arg in args]


def read_input(input_folder):
    X = pd.read_csv(f"{input_folder}/X.csv")
    y = pd.read_csv(f"{input_folder}/y.csv")

    categorical_columns = []
    if os.path.exists(f"{input_folder}/cat_features.txt"):
        with open(f"{input_folder}/cat_features.txt") as f:
            for line in f:
                if line.strip():
                    categorical_columns.append(line.strip())

    cat_features = None
    if categorical_columns:
        columns = X.columns
        cat_features = np.where(columns.isin(categorical_columns))[0]

        for col in list(columns[cat_features]):
            X[col] = X[col].astype(str)

    gs, _ = load_graphs(f"{input_folder}/graph.dgl")
    graph = gs[0]

    with open(f"{input_folder}/masks.json") as f:
        masks = json.load(f)

    return graph, X, y, cat_features, masks


def normalize_features(X, train_mask, val_mask, test_mask):
    min_max_scaler = preprocessing.MinMaxScaler()
    A = X.to_numpy(copy=True)
    A[train_mask] = min_max_scaler.fit_transform(A[train_mask])
    A[val_mask + test_mask] = min_max_scaler.transform(A[val_mask + test_mask])
    return pd.DataFrame(A, columns=X.columns).astype(float)


def replace_na(X, train_mask):
    if X.isna().any().any():
        return X.fillna(X.iloc[train_mask].min() - 1)
    return X


def encode_cat_features(X, y, cat_features, train_mask, val_mask, test_mask):
    enc = CatBoostEncoder()
    A = X.to_numpy(copy=True)
    b = y.to_numpy(copy=True)
    A[np.ix_(train_mask, cat_features)] = enc.fit_transform(
        A[np.ix_(train_mask, cat_features)], b[train_mask]
    )
    A[np.ix_(val_mask + test_mask, cat_features)] = enc.transform(
        A[np.ix_(val_mask + test_mask, cat_features)]
    )
    A = A.astype(float)
    return pd.DataFrame(A, columns=X.columns)


""" For CPF"""


class SparseGraph:
    """Attributed labeled graph stored in sparse matrix form."""

    def __init__(
        self,
        adj_matrix,
        attr_matrix=None,
        labels=None,
        node_names=None,
        attr_names=None,
        class_names=None,
        metadata=None,
    ):
        """Create an attributed graph.

        Parameters
        ----------
        adj_matrix : sp.csr_matrix, shape [num_nodes, num_nodes]
            Adjacency matrix in CSR format.
        attr_matrix : sp.csr_matrix or np.ndarray, shape [num_nodes, num_attr], optional
            Attribute matrix in CSR or numpy format.
        labels : np.ndarray, shape [num_nodes], optional
            Array, where each entry represents respective node's label(s).
        node_names : np.ndarray, shape [num_nodes], optional
            Names of nodes (as strings).
        attr_names : np.ndarray, shape [num_attr]
            Names of the attributes (as strings).
        class_names : np.ndarray, shape [num_classes], optional
            Names of the class labels (as strings).
        metadata : object
            Additional metadata such as text.

        """
        # Make sure that the dimensions of matrices / arrays all agree
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError(
                "Adjacency matrix must be in sparse format (got {0} instead)".format(
                    type(adj_matrix)
                )
            )

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError(
                    "Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)".format(
                        type(attr_matrix)
                    )
                )

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError(
                    "Dimensions of the adjacency and attribute matrices don't agree"
                )

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError(
                    "Dimensions of the adjacency matrix and the label vector don't agree"
                )

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError(
                    "Dimensions of the adjacency matrix and the node names don't agree"
                )

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError(
                    "Dimensions of the attribute matrix and the attribute names don't agree"
                )

        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels
        self.node_names = node_names
        self.attr_names = attr_names
        self.class_names = class_names
        self.metadata = metadata

    def num_nodes(self):
        """Get the number of nodes in the graph."""
        return self.adj_matrix.shape[0]

    def num_edges(self):
        """Get the number of edges in the graph.

        For undirected graphs, (i, j) and (j, i) are counted as single edge.
        """
        if self.is_directed():
            return int(self.adj_matrix.nnz)
        else:
            return int(self.adj_matrix.nnz / 2)

    def get_neighbors(self, idx):
        """Get the indices of neighbors of a given node.

        Parameters
        ----------
        idx : int
            Index of the node whose neighbors are of interest.

        """
        return self.adj_matrix[idx].indices

    def is_directed(self):
        """Check if the graph is directed (adjacency matrix is not symmetric)."""
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def to_undirected(self):
        """Convert to an undirected graph (make adjacency matrix symmetric)."""
        if self.is_weighted():
            raise ValueError("Convert to unweighted graph first.")
        else:
            self.adj_matrix = self.adj_matrix + self.adj_matrix.T
            self.adj_matrix[self.adj_matrix != 0] = 1
        return self

    def is_weighted(self):
        """Check if the graph is weighted (edge weights other than 1)."""
        return np.any(np.unique(self.adj_matrix[self.adj_matrix != 0].A1) != 1)

    def to_unweighted(self):
        """Convert to an unweighted graph (set all edge weights to 1)."""
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        return self

    # Quality of life (shortcuts)
    def standardize(self):
        """Select the LCC of the unweighted/undirected/no-self-loop graph.

        All changes are done inplace.

        """
        G = self.to_unweighted().to_undirected()
        G.adj_matrix = eliminate_self_loops_adj(G.adj_matrix)
        G = largest_connected_components(G, 1)
        return G

    def unpack(self):
        """Return the (A, X, z) triplet."""
        return self.adj_matrix, self.attr_matrix, self.labels


def load_npz_to_sparse_graph(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.

    """
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix(
            (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
            shape=loader["adj_shape"],
        )

        if "attr_data" in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix(
                (loader["attr_data"], loader["attr_indices"], loader["attr_indptr"]),
                shape=loader["attr_shape"],
            )
        elif "attr_matrix" in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader["attr_matrix"]
        else:
            attr_matrix = None

        if "labels_data" in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix(
                (
                    loader["labels_data"],
                    loader["labels_indices"],
                    loader["labels_indptr"],
                ),
                shape=loader["labels_shape"],
            )
        elif "labels" in loader:
            # Labels are stored as a numpy array
            labels = loader["labels"]
        else:
            labels = None

        node_names = loader.get("node_names")
        attr_names = loader.get("attr_names")
        class_names = loader.get("class_names")
        metadata = loader.get("metadata")

    return SparseGraph(
        adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata
    )


def sample_per_class(
    random_state, labels, num_examples_per_class, forbidden_indices=None
):
    """
    Used in get_train_val_test_split, when we try to get a fixed number of examples per class
    """

    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [
            random_state.choice(
                sample_indices_per_class[class_index],
                num_examples_per_class,
                replace=False,
            )
            for class_index in range(len(sample_indices_per_class))
        ]
    )


def get_train_val_test_split(
    random_state,
    labels,
    train_examples_per_class=None,
    val_examples_per_class=None,
    test_examples_per_class=None,
    train_size=None,
    val_size=None,
    test_size=None,
):

    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))
    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False
        )

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state,
            labels,
            val_examples_per_class,
            forbidden_indices=train_indices,
        )
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(
            random_state,
            labels,
            test_examples_per_class,
            forbidden_indices=forbidden_indices,
        )
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert (
            len(np.concatenate((train_indices, val_indices, test_indices)))
            == num_samples
        )

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices
