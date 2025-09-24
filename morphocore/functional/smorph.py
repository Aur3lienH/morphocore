import torch
from cpp_operation import smorph as smorph_cpp
from morphocore.functional.merge_enum import str_to_merge


def smorph(x: torch.Tensor, w: torch.Tensor, alpha: torch.Tensor, channel_merge_mode: str = "sum") -> torch.Tensor:
    """ 
    Function that approximate a dilation or an erosion depending on a alpha control parameter.

    Formula : (Needs reflection)

    Behaviour of alpha : 
        - When alpha -> ∞ then Smorph tends to be a dilation.
        - When alpha -> -∞ then Smorph tends to be an erosion.
        - When alpha is close to 0 -> then Smorph is something between an erosion and a dilation.
    """
    
    tanh_alpha = torch.tanh(alpha)
    term1 = (1 + tanh_alpha) / 2
    term2 = (1 - tanh_alpha) / 2

    w_tilde_alpha = tanh_alpha * (term1 * w.flip(dims=(-2, -1)) + term2 * w)
    return smorph_cpp(x, w_tilde_alpha, alpha, str_to_merge(channel_merge_mode))