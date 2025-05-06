from fastcore.utils import patch
import numpy
from pylot.torch.torchlib import torch

@patch
def np(tensor: torch.Tensor) -> numpy.ndarray:
    return tensor.detach().cpu().numpy()
