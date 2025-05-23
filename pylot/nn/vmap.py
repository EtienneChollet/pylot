from typing import Callable

from pylot.torch.torchlib import torch, nn
import einops as E

# Jaxlike vmap exploiting reshaping to the batch dimension.
# Applies the given module


def vmap(module: Callable, x: torch.Tensor, *args, **kwargs):
    batch_size, group_size, *_ = x.shape
    grouped_input = E.rearrange(x, "B S ... -> (B S) ...")
    grouped_output = module(grouped_input, *args, **kwargs)
    output = E.rearrange(
        grouped_output, "(B S) ... -> B S ...", B=batch_size, S=group_size
    )
    return output


def vmap_fn(fn: Callable):
    def vmapped_fn(*args, **kwargs):
        return vmap(fn, *args, **kwargs)

    return vmapped_fn


class Vmap(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor):
        return vmap(self.module, x)


def vmap_cls(module_type: type):
    def vmapped_cls(*args, **kwargs):
        module = module_type(*args, **kwargs)
        return Vmap(module)

    return vmapped_cls
