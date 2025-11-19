import torch
import torch.nn as nn
import torch.nn.functional as F


class MorphoModule(nn.Module):
    """
    Common module for Mathematical Morpholocial operations !
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_shape: tuple, channel_merge_mode: str = "sum", dtype: torch.dtype = torch.float32):
        super().__init__()
        if channel_merge_mode == "identity":
            if in_channels != out_channels:
                raise ValueError("For 'identity' channel_merge_mode, in_channels must be equal to out_channels.")
            self.weight = nn.Parameter(torch.zeros((out_channels, 1, *kernel_shape), dtype=dtype))
        else:
            self.weight = nn.Parameter(torch.zeros((out_channels, in_channels, *kernel_shape), dtype=dtype))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_merge_mode = channel_merge_mode