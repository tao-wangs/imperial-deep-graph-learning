import torch
import torch.nn as nn
import torch.nn.functional as F
from globals import TORCH_DEVICE
from initializations import *
from preprocessing import normalize_adj_torch


class GSRLayer(nn.Module):

    def __init__(self,hr_dim):
        super(GSRLayer, self).__init__()

        self.weights = torch.from_numpy(weight_variable_glorot(hr_dim)).to(TORCH_DEVICE, dtype=torch.float32)
        self.weights = torch.nn.Parameter(data=self.weights, requires_grad = True)

    def forward(self, adj_mx, input_features):
        lr = adj_mx
        lr_dim = lr.shape[0]
        # Get the eigen vectors of the LR graph
        _, U_lr = torch.linalg.eigh(lr, UPLO='U')
        U_lr = U_lr.to(TORCH_DEVICE, dtype=torch.float32)
        ident = torch.eye(lr_dim).to(TORCH_DEVICE, dtype=torch.float32)
        # Construct the S matrix, by concatenating 2 identity matrices
        s_d = torch.cat((ident, ident), 0)

        # Multiply the weight matrix with the S matrix and the
        # eigenvector matrix of LR to obtain the HR prediction
        connectivity_mx = torch.matmul(self.weights, s_d)
        b = torch.matmul(connectivity_mx ,torch.t(U_lr))
        f_d = torch.matmul(b, input_features)
        f_d = torch.abs(f_d)
        f_d = f_d.fill_diagonal_(1)
        normalized_adj_mx = normalize_adj_torch(f_d)
        input_features = torch.mm(normalized_adj_mx, normalized_adj_mx.t())
        input_features = (input_features + input_features.t()) / 2
        idx = torch.eye(320, device=TORCH_DEVICE, dtype=bool)
        input_features[idx] = 1
        return normalized_adj_mx, torch.abs(input_features)



class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, activation=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_dim = in_features
        self.out_dim = out_features
        self.projection = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, out_features),
        )
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.projection.named_parameters():
            if name == 'weight':
                torch.nn.init.xavier_uniform_(param)

    def forward(self, input_features, adj_mx):
        new_features = self.projection(input_features)

        return adj_mx @ new_features


class GraphConvolutionWithResidual(nn.Module):

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolutionWithResidual, self).__init__()

        self.conv1 = GraphConvolution(in_features, out_features, dropout, act)
        self.conv2 = GraphConvolution(out_features, in_features, dropout, act)

    def forward(self, input, adj):

        out = self.conv1(input, adj)
        out = F.relu(out)
        out = self.conv2(input, adj)
        out += input
        out = F.relu(out)

        return out
