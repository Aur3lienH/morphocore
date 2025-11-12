import torch
from torch import nn
from torch.functional import F
from morphocore.functional import dilation
from morphocore.functional import erosion

def emorph(x, w1, w2, alpha, channel_merge_mode, save_indices: bool = True):
    """
    Exact morphological function

    Formula : 
    The operation is Y(X) = sigmoid(alpha) * dilation(X, W) + (1.0 - sigmoid(alpha)) * erosion(X, W)
    where :
        - X is the input
        - W is the kernel
        - alpha is the control parameter between erosion and dilation

    Behaviour with different alpha : 
        - When alpha -> ∞ then Emorph tends to be a dilation 
        - When alpha -> -∞ then Emorph tends to be an erosion
        - When alpha is close to 0 -> then Emorph is something between an erosion and a dilation.
    
    Args:
        x (torch.Tensor): input
        w (torch.Tensor): weight
        alpha (torch.Tensor): control parameter
    """
    out_erosion = erosion(x, w1, channel_merge_mode=channel_merge_mode, save_indices=save_indices)
    out_dilation = dilation(x, w2, channel_merge_mode=channel_merge_mode, save_indices=save_indices)

    soft = torch.sigmoid(alpha.mean(dim=1))  # (C_out,)
    soft = soft.view(1, -1, 1, 1)  # (1, C_out, 1, 1)
    
    return out_erosion * (1.0 - soft) + out_dilation * soft