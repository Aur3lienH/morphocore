import torch
import torch.nn as nn
import torch.nn.functional as F


class MorphoModule(nn.Module):
    """
    Common module for Mathematical Morpholocial operations !
    """
    def __init__(self, in_channel: int, out_channel: int, kernel_shape: tuple, channel_merge_mode: str = "sum", dtype: torch.dtype = torch.float32):
        super().__init__()
        fan_in = in_channel * kernel_shape[0] * kernel_shape[1]
        std = (1.0 / fan_in) ** 0.5
        bound = std * (3 ** 0.5)
        self.weight = nn.Parameter(torch.empty((out_channel, in_channel, *kernel_shape), dtype=dtype).uniform_(-bound, bound))
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.channel_merge_mode = channel_merge_mode