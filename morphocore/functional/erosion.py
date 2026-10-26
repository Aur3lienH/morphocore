from morphocore.functional._ops import ops
from morphocore.functional.merge_enum import str_to_merge, str_to_padding
import torch

def erosion(x: torch.Tensor, w: torch.Tensor, channel_merge_mode: str = 'sum',
            padding_mode: str = 'replicate', save_indices: bool = True,
            weight_dropout: float = 0.0, training: bool = False,
            padding_value: float = 0.0, noise_scale: float = 0.0) -> torch.Tensor:
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

        - 'zero': border pixels read as 0.
        - 'geodesic': border pixels read as +inf, so they never win the minimum.
        - 'constant': border pixels read as `padding_value`.
        - 'replicate': border pixels read as the nearest edge pixel.

        Default is 'replicate'.
    save_indices : bool, optional
        Whether to save the indices of minimum values for gradient computation if the implementation exists for the specific configuration.
        Disabling it may reduce memory usage.
        Enabling it is recommended for performance when training.
        Default is True.
    weight_dropout : float, optional
        Probability of dropping weights in the structuring element during training.
        Dropped weights are set to -inf so they never contribute to the minimum.
        Must be in range [0, 1). Default is 0.0 (no dropout).
    training : bool, optional
        Whether the model is in training mode. Dropout and noise are only applied when True.
        Default is False.
    padding_value : float, optional
        Border value used when padding_mode is 'constant'. Ignored by the other
        padding modes. Default is 0.0.
    noise_scale : float, optional
        Half width of the uniform noise added to the structuring element during training.
        Each batch element draws its own noise in [-noise_scale, noise_scale].
        Default is 0.0 (no noise).

    Returns
    -------
    torch.Tensor
        Eroded output tensor of shape (N, C_out, H, W).

    Notes
    -----
    The operation is defined as: (x ⊖ w)[i,j] = min{x[i+m, j+n] - w[m,n]}.
    When weight_dropout is applied, dropped weights are set to -inf, ensuring
    x - (-inf) = +inf and they are never selected as the minimum.
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
    >>> # With dropout during training
    >>> output = erosion(x, w, weight_dropout=0.2, training=True)
    >>> # Padding the border with a fixed value
    >>> output = erosion(x, w, padding_mode='constant', padding_value=1.0)
    """
    # Apply weight dropout if in training mode
    if training and weight_dropout > 0.0:
        dropout_mask = torch.rand_like(w) > weight_dropout
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                if not dropout_mask[i, j].any():
                    flat_idx = torch.randint(0, dropout_mask[i, j].numel(), (1,)).item()
                    dropout_mask[i, j].view(-1)[flat_idx] = True
        
        w = torch.where(dropout_mask, w, torch.tensor(-float('inf'), device=w.device, dtype=w.dtype))
    
    return ops.erosion(x, w.flip((-2, -1)), str_to_merge(channel_merge_mode),
                       str_to_padding(padding_mode), padding_value, save_indices,
                       noise_scale if training else 0.0)