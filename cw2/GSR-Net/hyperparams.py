from dataclasses import dataclass

import torch


@dataclass
class Hyperparams():
    epochs: int = 100
    lr: float = 0.005
    lr_schedule: float = 0.98
    splits: int = 3
    lmbda: float = 5.0
    lr_dim: int = 160
    hr_dim: int = 320
    hidden_dim: int = 320
    padding: int = 26
    num_ks: int = 5
    k_multiplier: float = 0.85
    dropout: float = 0.1
    num_residual_conv: int = 2
    train_criterion: torch.nn.modules.loss = torch.nn.MSELoss()
