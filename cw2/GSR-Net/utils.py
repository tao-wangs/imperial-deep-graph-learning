import torch
import torch.nn.functional as F

# def initialise_embeddings_identity(embeddings):
#     return 1 * embeddings

# def initialise_embeddings_random(embeddings):
#     m, n = embeddings.shape
#     return torch.randn(m, n)

# def normalize_adj(adjacency):
#     epsilon = 1e-5 # Small constant to avoid division by zero
#     adjacency = adjacency + torch.eye(adjacency.size(0)) # add self-connections
#     inv_degree_mx = torch.diag(1.0 / (epsilon + torch.sum(adjacency, dim=0)))
#     adj_hat = inv_degree_mx.matmul(adjacency)
#     return adj_hat

def pad_HR_adj(label, split):
    label = F.pad(label, (split, split, split, split), "constant", 0)
    # for idx in range(label.shape[0]):
    # label[idx] = label[idx].fill_diagonal_(1)
    label = label.fill_diagonal_(1)
    return label.to(dtype=torch.float32)

def unpad(data, split):

  idx_0 = data.shape[0]-split
  idx_1 = data.shape[1]-split
  # print(idx_0,idx_1)
  train = data[split:idx_0, split:idx_1]
  return train
