import torch
import torch.nn as nn
import torch.nn.functional as F
from globals import TORCH_DEVICE


class GraphUnpool(nn.Module):

    def __init__(self):
        super(GraphUnpool, self).__init__()

    def forward(self, adj_mx, input_features, idx):
        new_features = torch.zeros([adj_mx.shape[0], input_features.shape[1]]).to(TORCH_DEVICE)
        new_features[idx] = input_features
        return adj_mx, new_features


class GraphPool(nn.Module):

    def __init__(self, k, in_dim):
        super(GraphPool, self).__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, adj_mx, input_features):
        scores = self.proj(input_features)
        scores = torch.squeeze(scores)
        scores = self.sigmoid(scores / 100)
        num_nodes = adj_mx.shape[0]
        values, idx = torch.topk(scores, int(self.k * num_nodes))
        new_features = input_features[idx, :]
        values = torch.unsqueeze(values, -1)
        new_features = torch.mul(new_features, values)
        adj_mx = adj_mx[idx, :]
        adj_mx = adj_mx[:, idx]
        return adj_mx, new_features, idx

class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(GAT, self).__init__()
        self.projection = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, out_dim),
        )
        self.out_dim = out_dim
        self.phi = torch.nn.Parameter(torch.randn(2 * out_dim, 1).to(TORCH_DEVICE, dtype=torch.float32))
        self.activation = nn.LeakyReLU()

    def forward(self, adj_mx, input_features):
        num_nodes = adj_mx.shape[0]
        new_features = self.projection(input_features)
        first_half = (new_features @ self.phi[:self.out_dim]).view(-1, 1)
        second_half = (new_features @ self.phi[self.out_dim:]).view(1, -1)
        similarity_mx = self.activation(first_half + second_half)

        adj_hat = adj_mx + torch.eye(num_nodes).to(TORCH_DEVICE)
        masked_s = torch.where(adj_hat > 0, similarity_mx, -1e20).to(TORCH_DEVICE)
        masked_similarity_mx = F.softmax(masked_s, dim=1).to(TORCH_DEVICE)
        new_features_with_attention = masked_similarity_mx @ new_features

        return self.activation(new_features_with_attention)


class GraphUnet(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dropout=0, dim=320):
        super(GraphUnet, self).__init__()
        self.ks = ks

        # replaced every GCN layer with GAT layers
        self.start_gcn = GAT(in_dim, dim, dropout=dropout)
        self.latent_gcn = GAT(dim, dim, dropout=dropout)
        self.end_gcn = GAT(2 * dim, out_dim, dropout=dropout)
        self.pool_level_gcn = []
        self.unpool_level_gcn = []
        self.pools = []
        self.unpools = []
        self.l_n = len(ks)
        for i in range(self.l_n):
            # GAT layers applied before any 2 pooling/unpooling layers
            self.pool_level_gcn.append(GAT(dim, dim, dropout=dropout).to(TORCH_DEVICE))
            self.unpool_level_gcn.append(GAT(dim, dim, dropout=dropout).to(TORCH_DEVICE))
            self.pools.append(GraphPool(ks[i], dim).to(TORCH_DEVICE))
            self.unpools.append(GraphUnpool().to(TORCH_DEVICE))

    def forward(self, adj_mx, input_features):
        adj_matrices = []
        indices_list = []
        down_outs = []
        # Apply the initial GAT layer
        input_features = self.start_gcn(adj_mx, input_features)
        start_gcn_outs = input_features
        original_input = input_features
        # Apply the pooling layers
        for i in range(self.l_n):
            input_features = self.pool_level_gcn[i](adj_mx, input_features)
            adj_matrices.append(adj_mx)
            down_outs.append(input_features)
            adj_mx, input_features, idx = self.pools[i](adj_mx, input_features)
            indices_list.append(idx)
        input_features = self.latent_gcn(adj_mx, input_features)
        # Apply the unpooling layers
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            adj_mx, idx = adj_matrices[up_idx], indices_list[up_idx]
            adj_mx, input_features = self.unpools[i](adj_mx, input_features, idx)
            input_features = self.unpool_level_gcn[i](adj_mx, input_features)
            input_features = input_features.add(down_outs[up_idx])
        # Concatenate the initial features (after the first conv layer) and
        # the decoded features to obtain a lr_dim x hr_dim node embeddings matrix
        input_features = torch.cat([input_features, original_input], 1)
        input_features = self.end_gcn(adj_mx, input_features)

        return input_features, start_gcn_outs
