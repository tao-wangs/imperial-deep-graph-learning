import torch
import torch.nn as nn
import torch.nn.functional as F
from globals import TORCH_DEVICE
from layers import *
from ops import *
from preprocessing import normalize_adj_torch


class GSRNet(nn.Module):

    def __init__(self, hps):
        super(GSRNet, self).__init__()
        # results after several layers
        self.net_outs, self.start_gcn_outs = None, None
        self.outputs, self.Z = None, None
        self.hidden1, self.hidden2 = None, None
        self.last_hidden = None

        # pooling/unpooling multipliers between consecutive layers
        self.ks = []
        for i in range(hps.num_ks):
            self.ks.append(hps.k_multiplier ** i)

        self.lr_dim = hps.lr_dim
        self.hr_dim = hps.hr_dim
        self.hidden_dim = hps.hidden_dim
        self.layer = GSRLayer(self.hr_dim)
        self.net = GraphUnet(self.ks, self.lr_dim, self.hr_dim)
        self.num_residual_conv = hps.num_residual_conv
        for i in range(self.num_residual_conv):
            setattr(self, f"gc{i}", GraphConvolutionWithResidual(self.hr_dim, self.hidden_dim, hps.dropout, act=F.relu))

    def forward(self, lr):

        ident = torch.eye(self.lr_dim).to(TORCH_DEVICE, dtype=torch.float32)
        adj_mx = normalize_adj_torch(lr).to(TORCH_DEVICE, dtype=torch.float32)

        # Put the LR graph and the initial node features through the UNet
        # and get the auto-decoded LR graph node features 
        # and the graph features after the initial GAT layer
        self.net_outs, self.start_gcn_outs = self.net(adj_mx, ident)

        # Apply the super-resolution layer
        self.outputs, self.Z = self.layer(adj_mx, self.net_outs)

        # Apply the final residual convolutional layers
        self.last_hidden = self.Z
        for i in range(self.num_residual_conv):
            self.last_hidden = getattr(self, f"gc{i}")(self.last_hidden, self.outputs)

        # Make sure the adjacency matrix symmetric
        z = self.last_hidden
        z = (z + z.t()) / 2
        idx = torch.eye(self.hr_dim, dtype=bool)
        z[idx] = 1

        return torch.abs(z), self.net_outs, self.start_gcn_outs, self.outputs
