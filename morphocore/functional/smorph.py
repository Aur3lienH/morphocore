import torch
from cpp_operation import smorph as smorph_cpp
from cpp_operation import smorph_scm as smorph_scm_cpp
from morphocore.functional.merge_enum import str_to_merge


def smorph(x: torch.Tensor, w: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor = None, channel_merge_mode: str = "sum") -> torch.Tensor:
    """
    Apply smooth morphological operation with learnable interpolation parameter.
    
    Approximates a morphological operation that smoothly transitions from erosion to dilation
    based on the alpha parameter. When alpha → +∞, the operation becomes dilation; 
    when alpha → -∞, it becomes erosion.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (N, C_in, H, W) where N is batch size, C_in is the number
        of input channels, H is height, and W is width.
    w : torch.Tensor
        Structuring element (kernel) of shape (C_out, C_in, K_h, K_w) where C_out is the
        number of output channels, K_h and K_w are the kernel height and width.
    alpha : torch.Tensor
        Control parameter of shape (C_out, C_in) or (C_out,) that interpolates between 
        erosion and dilation. When alpha → +∞, output → dilation; when alpha → -∞, 
        output → erosion.
    beta : torch.Tensor, optional
        Additional control parameter for selective channel merging. Default is None.
    channel_merge_mode : str, optional
        Strategy for merging multiple input channels. Options include:
        'sum', 'max', 'min', 'average' or 'identity'. Default is 'sum'.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (N, C_out, H, W).

    Notes
    -----
    The operation uses a smooth approximation with tanh activation:
    w̃(α) = tanh(α) × [(1-tanh(α))/2 × w_flipped + (1+tanh(α))/2 × w]
    
    Supports GPU acceleration via CUDA and backpropagation for trainable layers.

    Examples
    --------
    >>> import torch
    >>> from morphocore.functional import smorph
    >>> x = torch.randn(1, 3, 32, 32)
    >>> w = torch.ones(3, 3, 5, 5)
    >>> alpha = torch.zeros(3, 3)  # Balanced between erosion and dilation
    >>> output = smorph(x, w, alpha)
    >>> output.shape
    torch.Size([1, 3, 32, 32])
    """
    
    if channel_merge_mode == 'identity':
        if alpha.dim() == 1:
            tanh_alpha = torch.tanh(alpha).view(-1, 1, 1, 1)
        else:
            tanh_alpha = torch.tanh(alpha).view(-1, 1, 1, 1)
    else:
        if alpha.dim() == 1:
            tanh_alpha = torch.tanh(alpha).view(-1, 1, 1, 1)
        else:
            tanh_alpha = torch.tanh(alpha).unsqueeze(-1).unsqueeze(-1)
    
    term1 = (1 + tanh_alpha) / 2
    term2 = (1 - tanh_alpha) / 2

    w_tilde_alpha = tanh_alpha * (term2 * w.flip(dims=(-2, -1)) + term1 * w)

    if beta is not None:
        return smorph_scm_cpp(x, w_tilde_alpha, alpha, beta, str_to_merge(channel_merge_mode))
    
    return smorph_cpp(x, w_tilde_alpha, alpha, str_to_merge(channel_merge_mode))