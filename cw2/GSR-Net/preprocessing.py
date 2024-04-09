import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from datasets import BrainDataset
from MatrixVectorizer import MatrixVectorizer
# from utils import normalize_adj

def get_dset():
    # Prepare low + high resolution training set
    lr_data = pd.read_csv('data/lr_train.csv')
    hr_data = pd.read_csv('data/hr_train.csv')

    # Prepare training dataset
    vectorizer = MatrixVectorizer()

    n_samples = len(lr_data)
    train_dataset = []

    lr_train_vectorized = np.array(lr_data)
    hr_train_vectorized = np.array(hr_data)

    for i in range(n_samples):
        lr_train_matrix = vectorizer.anti_vectorize(lr_train_vectorized[i], NUM_LOW_RES_NODES)
        hr_train_matrix = vectorizer.anti_vectorize(hr_train_vectorized[i], NUM_HIGH_RES_NODES)
        # lr_train_feature_matrix = init_feature_matrix(NUM_LOW_RES_NODES, NUM_FEATURES_PER_NODE)
        train_dataset.append((lr_train_matrix, hr_train_matrix))

    return train_dataset

def get_unseen_test_dset():
    lr_data = pd.read_csv('data/lr_test.csv')
    vectorizer = MatrixVectorizer()
    n_samples = len(lr_data)
    test_dataset = []
    lr_test_vectorized = np.array(lr_data)

    for i in range(n_samples):
        lr_test_matrix = vectorizer.anti_vectorize(lr_test_vectorized[i], NUM_LOW_RES_NODES)
        # lr_test_feature_matrix = init_feature_matrix(NUM_LOW_RES_NODES, NUM_FEATURES_PER_NODE)
        # test_dataset.append((lr_test_matrix, lr_test_feature_matrix))
        test_dataset.append(lr_test_matrix)

    return test_dataset

def get_dataloaders(dset, hps, is_train_or_val=True):
    # temporary
    normalization_func = normalize_adj
    batch_size = hps.batch_size

    if is_train_or_val:
        train_dataset, val_dataset = train_test_split(dset, test_size=0.2, random_state=RANDOM_SEED)
        train_dataset = BrainDataset(train_dataset, normalization_func=normalization_func)
        val_dataset = BrainDataset(val_dataset, normalization_func=normalization_func)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
    else:
        test_dataset = BrainDataset(dset, normalization_func=normalization_func, is_train_or_val=is_train_or_val)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return test_loader

# def init_feature_matrix(num_nodes, num_features):
#     # Initialize feature matrix - by the paper, it needs to be initialized to the identity matrix
#     # ----------- maybe normal init might be an improvement ------------
#     # feature_matrix = np.empty((num_nodes, num_features))
#     # for i in range(num_nodes):
#     #     feature_matrix[i] = np.random.normal(0, 1, num_features)

#     feature_matrix = np.eye(num_nodes, num_nodes)
#     return feature_matrix

def normalize_adj_torch(mx):
    # mx = mx.to_dense()
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    return mx

NUM_FEATURES_PER_NODE = 5
RANDOM_SEED = 42
NUM_LOW_RES_NODES = 160
NUM_HIGH_RES_NODES = 268
