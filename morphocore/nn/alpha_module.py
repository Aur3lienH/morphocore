import torch
import torch.nn as nn
import torch.nn.functional as F
from .morpho_module import MorphoModule


class AlphaModule(MorphoModule):
    """
    Common module for Mathematical Morpholocial operations which approximate both erosion and dilation depending on alpha !
    """
    def __init__(self, in_channel : int, out_channel : int, kernel_shape : tuple, alpha : float = 0.0):
        super().__init__(in_channel, out_channel, kernel_shape)
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float))


        