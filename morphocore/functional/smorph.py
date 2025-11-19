import torch
from cpp_operation import smorph as smorph_cpp
from cpp_operation import smorph_scm as smorph_scm_cpp
from morphocore.functional.merge_enum import str_to_merge


def smorph(x: torch.Tensor, w: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor = None, channel_merge_mode: str = "sum") -> torch.Tensor:
    """ 
    Function that approximates a dilation or an erosion depending on an alpha control parameter.

    Formula : (Needs reflection)

    Behaviour of alpha : 
        - When alpha -> ∞ then Smorph tends to be a dilation.
        - When alpha -> -∞ then Smorph tends to be an erosion.
        - When alpha is close to 0 -> then Smorph is something between an erosion and a dilation.
    
    Args:
        x: Input tensor of shape [batch, in_channels, height, width]
        w: Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width]
        alpha: Control parameter tensor of shape [out_channels, in_channels]
        channel_merge_mode: How to merge across input channels ("sum", "max", or "mean")
    """

    
    tanh_alpha = torch.tanh(alpha).unsqueeze(-1).unsqueeze(-1) 
    
    term1 = (1 + tanh_alpha) / 2
    term2 = (1 - tanh_alpha) / 2

    w_tilde_alpha = tanh_alpha * (term2 * w.flip(dims=(-2, -1)) + term1 * w)

    if beta is not None:
        return smorph_scm_cpp(x, w_tilde_alpha, alpha, beta, str_to_merge(channel_merge_mode))
    
    return smorph_cpp(x, w_tilde_alpha, alpha, str_to_merge(channel_merge_mode))