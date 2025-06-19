import random
import os
import numpy as np
import torch

def set_seed(seed: int):
    """
    Sets seed across random, numpy, torch (CPU & GPU) for reproducibility.

    Parameters:
    -----------
    seed : int
        The random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False