import random

import numpy as np
import torch
from preprocessing import RANDOM_SEED, get_dset


def setup_device():
    # Set a fixed random seed for reproducibility across multiple libraries
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Check for CUDA (GPU support) and set device accordingly
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)  # For multi-GPU setups
        # Additional settings for ensuring reproducibility on CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")

    return device

TORCH_DEVICE = setup_device()
BEST_LOSS_FOLD1, BEST_LOSS_FOLD2, BEST_LOSS_FOLD3 = 1e10, 1e10, 1e10
PATH_FOLD1, PATH_FOLD2, PATH_FOLD3 = "models/GSR_model_fold1.pth", "models/GSR_model_fold2.pth", "models/GSR_model_fold3.pth"

FULL_DATA, FULL_TARGETS = zip(*(get_dset()))
FULL_DATA = np.stack(FULL_DATA)
FULL_TARGETS = np.stack(FULL_TARGETS)

COMPUTE_METRICS_FLAG = True
