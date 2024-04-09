import numpy as np
import torch
import wandb
from globals import (BEST_LOSS_FOLD1, BEST_LOSS_FOLD2, BEST_LOSS_FOLD3,
                     FULL_DATA, FULL_TARGETS, PATH_FOLD1, PATH_FOLD2,
                     PATH_FOLD3, TORCH_DEVICE)
from hyperparams import Hyperparams
from model import GSRNet
from sklearn.model_selection import KFold
from train import test, train


def gsr_search_config(test_indices, best_loss, path) -> dict:
    train_index, test_index = test_indices
    search_config = {
        'method': 'bayes',
        'metric': {
            'name': 'mean_mae',
            'goal': 'minimize'
        }
    }
    search_config_params = {
        "num_epochs": {
            'values': [150]
        },
        "lr": {
            'values':  [0.0001, 0.0003, 0.0006, 0.001, 0.005, 0.01]
        },
        "lr_schedule": {
            'values': [1, 0.995, 0.99, 0.985, 0.98]
        },
        "num_ks": {
            'values': [2, 3, 4, 5]
        },
        "k_multiplier": {
            'values': [0.9, 0.85, 0.8, 0.75]
        },
        "lmbda": {
            'values': [1, 5, 10, 15, 20, 25]
        },
        "dropout": {
            'values': [0, 0.1, 0.15, 0.2, 0.25, 0.3]
        },
        "num_residual_conv": {
            'values': [1, 2, 3, 4]
        },
        "train_indices": {
            'values': [train_index.tolist()]
        },
        "test_indices": {
            'values': [test_index.tolist()]
        },
        "best_loss": {
            'values': [best_loss]
        },
        "path": {
            'values': [path]
        }
    }
    search_config['parameters'] = search_config_params

    return search_config

def gsr_grid_search():
    with wandb.init(resume=True):
        grid_config = wandb.config
        train_index, test_index = np.array(grid_config.train_indices), np.array(grid_config.test_indices)
        train_adj, test_adj = FULL_DATA[train_index], FULL_DATA[test_index]
        train_labels, test_labels = FULL_TARGETS[train_index], FULL_TARGETS[test_index]
        print("shapes" , train_adj.shape, test_adj.shape, train_labels.shape, test_labels.shape)
        best_loss, path = grid_config.best_loss, grid_config.path
        hps = Hyperparams(
            epochs = grid_config.num_epochs,
            lr = grid_config.lr,
            lr_schedule = grid_config.lr_schedule,
            num_ks = grid_config.num_ks,
            k_multiplier = grid_config.k_multiplier,
            lmbda = grid_config.lmbda,
            dropout = grid_config.dropout,
            num_residual_conv=grid_config.num_residual_conv
        )
        model = GSRNet(hps).to(TORCH_DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=hps.lr)
        train(model, optimizer, train_adj, train_labels, hps)
        current_mae_loss = test(model, test_adj, test_labels, hps)
        if current_mae_loss < best_loss:
            best_loss = current_mae_loss
            torch.save(model, path)


def setup():
    wandb.login(key="6af656612e6115c4b189c6074dadbfc436f21439")

def run_gsr_sweep():
    cv = KFold(n_splits=3, random_state=42, shuffle=True)
    split_indices = list(cv.split(FULL_DATA, FULL_TARGETS))

    setup()
    sweep_config_fold1 = gsr_search_config(split_indices[0], PATH_FOLD1, BEST_LOSS_FOLD1)
    sweep_config_fold2 = gsr_search_config(split_indices[1], PATH_FOLD2, BEST_LOSS_FOLD2)
    sweep_config_fold3 = gsr_search_config(split_indices[2], PATH_FOLD3, BEST_LOSS_FOLD3)

    sweep_id_fold1 = wandb.sweep(sweep_config_fold1, entity="dgbl", project="bayes_resnet1")
    sweep_id_fold2 = wandb.sweep(sweep_config_fold2, entity="dgbl", project="bayes_resnet2")
    sweep_id_fold3 = wandb.sweep(sweep_config_fold3, entity="dgbl", project="bayes_resnet3")
    wandb.agent(sweep_id=sweep_id_fold1, function=gsr_grid_search, count=75,
                entity="dgbl", project="bayes_resnet1")
    wandb.agent(sweep_id=sweep_id_fold2, function=gsr_grid_search, count=75,
                entity="dgbl", project="bayes_resnet2")
    wandb.agent(sweep_id=sweep_id_fold3, function=gsr_grid_search, count=75,
                entity="dgbl", project="bayes_resnet3")

run_gsr_sweep()
