import random

import torch
import numpy as np

def set_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)