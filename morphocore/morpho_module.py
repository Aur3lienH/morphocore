import torch
import torch.nn as nn
import torch.nn.functional as F


class MorphoModule(nn.Module):
    """
    Common module for Mathematical Morpholocial operations !
    """
    def __init__(self, in_channel: int, out_channel: int, kernel_shape: tuple):
        super().__init__()
        self.weight = nn.Parameter(torch.rand((in_channel,out_channel,*kernel_shape),dtype=torch.float))
        self.in_channel = in_channel
        self.out_channel = out_channel