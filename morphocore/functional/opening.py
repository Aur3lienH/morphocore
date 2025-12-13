import torch
from morphocore.functional import dilation, erosion


def opening(x: torch.Tensor, w_erosion: torch.Tensor, w_dilation: torch.Tensor, channel_merge_mode: str = 'sum', padding_mode: str = 'geodesic', save_indices: bool = True) -> torch.Tensor:
    """
    Apply morphological opening operation (erosion followed by dilation).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (N, C_in, H, W) where N is batch size, C_in is the number
        of input channels, H is height, and W is width.
    w_erosion : torch.Tensor
        Structuring element for erosion of shape (C_mid, C_in, K_h, K_w) where C_mid is the
        number of intermediate channels, K_h and K_w are the kernel height and width.
    w_dilation : torch.Tensor
        Structuring element for dilation of shape (C_out, C_mid, K_h, K_w) where C_out is the
        number of output channels. Note: C_out must equal C_mid since dilation uses identity
        channel merging to maintain one-to-one correspondence with erosion outputs.
    channel_merge_mode : str, optional
        Strategy for merging multiple input channels in the erosion step. Options include:
        'sum', 'max', 'min', 'average' or 'identity'. Default is 'sum'.
        Note: The dilation step always uses 'identity' mode.
    padding_mode : str, optional
        Padding strategy to handle border pixels. Options include:
        'geodesic' or 'replicate'. Default is 'geodesic'.
    save_indices : bool, optional
        Whether to save the indices of extremum values for gradient computation.
        Disabling it may reduce memory usage.
        Enabling it is recommended for performance when training.
        Default is True.

    Returns
    -------
    torch.Tensor
        Output tensor after opening operation of shape (N, C_out, H, W).

    Notes
    -----
    The opening operation is defined as: (x ○ w) = ((x ⊖ w_e) ⊕ w_d)
    where ⊖ is erosion and ⊕ is dilation.
    
    The dilation step uses 'identity' channel merge mode to ensure each eroded channel
    is dilated independently, maintaining the morphological property where each erosion
    is paired with its corresponding dilation.
    
    Opening tends to:
    - Remove small objects and noise
    - Separate touching objects
    - Smooth object boundaries from the inside
    - Preserve the overall size and shape better than individual operations
    
    Supports GPU acceleration via CUDA and backpropagation for trainable layers.

    Examples
    --------
    Basic usage with same structuring element for both operations:
    
    >>> import torch
    >>> from morphocore.functional import opening
    >>> x = torch.randn(1, 3, 32, 32)
    >>> w = torch.ones(3, 3, 5, 5)
    >>> output = opening(x, w, w)
    >>> output.shape
    torch.Size([1, 3, 32, 32])
    
    Using different structuring elements with intermediate channels:
    
    >>> w_erosion = torch.ones(16, 3, 5, 5)   # 3 -> 16 channels
    >>> w_dilation = torch.ones(16, 16, 5, 5)   # 16 -> 16 channels (identity merge)
    >>> output = opening(x, w_erosion, w_dilation, channel_merge_mode='max')
    >>> output.shape
    torch.Size([1, 16, 32, 32])
    
    See Also
    --------
    closing : Morphological closing (dilation followed by erosion)
    dilation : Morphological dilation operation
    erosion : Morphological erosion operation
    """
    eroded = erosion(x, w_erosion, channel_merge_mode, padding_mode, save_indices)
    return dilation(eroded, w_dilation, 'identity', padding_mode, save_indices)