import os
import random
import numpy as np
import torch

def set_seed(seed: int):
    """Set seed for reproducibility across random, NumPy, and PyTorch (CPU & GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def schedule_epoch_lambda(epoch, lambda_0, max_times=1., min_times=1., 
                          n_epoch_warmup=0, n_epoch_cooldown=0):

    if epoch < n_epoch_warmup:
        return lambda_0  + lambda_0 * (max_times - 1) * epoch / n_epoch_warmup
        
    elif epoch < n_epoch_warmup + n_epoch_cooldown:
        return lambda_0 * max_times - lambda_0 * (max_times - min_times) * (epoch - n_epoch_warmup) / n_epoch_cooldown
        
    else:
        return lambda_0 * min_times
