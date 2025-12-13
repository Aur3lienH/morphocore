from cpp_operation import morpho_erosion
from morphocore.functional.merge_enum import str_to_merge, str_to_padding
import torch

def erosion(x: torch.Tensor, w: torch.Tensor, channel_merge_mode: str = 'sum', padding_mode: str = 'geodesic', save_indices: bool = True) -> torch.Tensor:
    """
    Apply morphological erosion operation on input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (N, C_in, H, W) where N is batch size, C_in is the number
        of input channels, H is height, and W is width.
    w : torch.Tensor
        Structuring element (kernel) of shape (C_out, C_in, K_h, K_w) where C_out is the
        number of output channels, K_h and K_w are the kernel height and width respectively.
    channel_merge_mode : str, optional
        Strategy for merging multiple input channels. Options include:
        'sum', 'max', 'min', 'average' or 'identity'. Default is 'sum'.
    padding_mode : str, optional
        Padding strategy to handle border pixels. Options include:
        'geodesic' or 'replicate'. Default is 'geodesic'.
    save_indices : bool, optional
        Whether to save the indices of minimum values for gradient computation if the implementation exists for the specific configuration.
        Disabling it may reduce memory usage.
        Enabling it is recommended for performance when training.
        Default is True.

    Returns
    -------
    torch.Tensor
        Eroded output tensor of shape (N, C_out, H, W).

    Notes
    -----
    The operation is defined as: (x ⊖ w)[i,j] = min{x[i+m, j+n] - w[m,n]}.
    Supports GPU acceleration via CUDA and backpropagation for trainable layers.

    Examples
    --------
    >>> import torch
    >>> from morphocore.functional import erosion
    >>> x = torch.randn(1, 3, 32, 32)
    >>> w = torch.ones(3, 3, 5, 5)
    >>> output = erosion(x, w, channel_merge_mode='max')
    >>> output.shape
    torch.Size([1, 3, 32, 32])
    """
    return morpho_erosion(x, w.flip((-2, -1)), str_to_merge(channel_merge_mode), str_to_padding(padding_mode), save_indices)