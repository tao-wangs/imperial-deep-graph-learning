import torch
from torch.utils.data import Dataset


class BrainDataset(Dataset):
    def __init__(self, dataset, normalization_func=None, is_train_or_val=True):
        self.dataset = dataset
        self.normalization_func = normalization_func
        self.seen = is_train_or_val

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        elem = None
        if self.seen:
            lr_matrix, lr_feature_matrix, hr_matrix = self.dataset[index]
            elem = (torch.from_numpy(lr_matrix), torch.from_numpy(lr_feature_matrix), torch.from_numpy(hr_matrix))
        else:
            lr_matrix, lr_feature_matrix = self.dataset[index]
            elem = (torch.from_numpy(lr_matrix), torch.from_numpy(lr_feature_matrix))

        return elem