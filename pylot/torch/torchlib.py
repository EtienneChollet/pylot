"""
'barrel imports' for pytorch
"""

__all__ = ["torch", "nn", "Tensor", "F", "Dataset", "DataLoader"]

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
