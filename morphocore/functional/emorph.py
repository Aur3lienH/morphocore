import torch
from torch import nn
from torch.functional import F
from morphocore.functional import dilation
from morphocore.functional import erosion

def emorph(x: torch.Tensor, w: torch.Tensor, alpha: torch.Tensor, channel_merge_mode: str = 'sum', padding_mode: str = 'geodesic', save_indices: bool = True) -> torch.Tensor:
    """
    Apply morphological interpolation between erosion and dilation.
    
    The operation smoothly transitions from erosion to dilation based on the alpha parameter:
    when alpha → +∞, the operation becomes dilation; when alpha → -∞, it becomes erosion.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (N, C_in, H, W) where N is batch size, C_in is the number
        of input channels, H is height, and W is width.
    w : torch.Tensor
        Structuring element (kernel) of shape (C_out, C_in, K_h, K_w) where C_out is the
        number of output channels, K_h and K_w are the kernel height and width.
    alpha : torch.Tensor
        Control parameter of shape (C_out, C_in) that interpolates between erosion and dilation.
        When alpha → +∞, output → dilation; when alpha → -∞, output → erosion.
    channel_merge_mode : str, optional
        Strategy for merging multiple input channels. Options include:
        'sum', 'max', 'min', 'average' or 'identity'. Default is 'sum'.
    padding_mode : str, optional
        Padding strategy to handle border pixels. Options include:
        'geodesic' or 'replicate'. Default is 'geodesic'.
    save_indices : bool, optional
        Whether to save the indices of extremum values for gradient computation if the implementation exists for the specific configuration.
        Disabling it may reduce memory usage.
        Enabling it is recommended for performance when training.
        Default is True.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (N, C_out, H, W).

    Notes
    -----
    The operation is defined as: Y(X) = σ(α) ⊕ X + (1 - σ(α)) ⊖ X
    where σ is the sigmoid function, ⊕ is dilation, and ⊖ is erosion.
    
    Supports GPU acceleration via CUDA and backpropagation for trainable layers.

    Examples
    --------
    >>> import torch
    >>> from morphocore.functional import emorph
    >>> x = torch.randn(1, 3, 32, 32)
    >>> w = torch.ones(3, 3, 5, 5)
    >>> alpha = torch.zeros(3, 3)  # Balanced between erosion and dilation
    >>> output = emorph(x, w, alpha)
    >>> output.shape
    torch.Size([1, 3, 32, 32])
    """
    out_erosion = erosion(x, w, channel_merge_mode=channel_merge_mode, padding_mode=padding_mode, save_indices=save_indices)
    out_dilation = dilation(x, w, channel_merge_mode=channel_merge_mode, padding_mode=padding_mode, save_indices=save_indices)

    soft = torch.sigmoid(alpha.mean(dim=1))  # (C_out,)
    soft = soft.view(1, -1, 1, 1)  # (1, C_out, 1, 1)
    
    return out_erosion * (1.0 - soft) + out_dilation * soft