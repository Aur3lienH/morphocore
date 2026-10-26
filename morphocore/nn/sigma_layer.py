import torch
from torch import nn
from .alpha_module import AlphaModule
from morphocore.functional import sigma_layer


class SigmaLayer(AlphaModule):
    """
    Learnable morphological layer with exact interpolation between erosion and dilation.
    
    This layer smoothly transitions from erosion to dilation based on a learnable alpha parameter.
    When alpha → +∞, the operation becomes dilation; when alpha → -∞, it becomes erosion.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple of int
        Size of the structuring element. If int, a square kernel is used.
    channel_merge_mode : str, optional
        Strategy for merging multiple input channels. Options include:
        'sum', 'max', 'min', 'average' or 'identity'. Default is 'sum'.
    padding_mode : str, optional
        Padding strategy to handle border pixels. Options include:
        'zero', 'geodesic', 'constant' or 'replicate'. Default is 'replicate'.
    init_alpha : float, optional
        Initial value for the alpha parameter. Default is 0.0.
    dtype : torch.dtype, optional
        Data type for the weight and alpha parameters. Default is torch.float32.
    weight_dropout : float, optional
        Probability of dropping weights in the structuring element during training.
        Each stage draws its own mask. Must be in range [0, 1). Default is 0.0 (no dropout).
    noise_scale : float, optional
        Half width of the uniform noise added to the structuring element during training.
        Each stage and batch element draws its own noise in [-noise_scale, noise_scale].
        Only supported on CPU. Default is 0.0 (no noise).
    save_indices : bool, optional
        Whether to save the indices of extremum values for gradient computation.
        Default is True.

    Attributes
    ----------
    weight : torch.nn.Parameter
        Learnable structuring element of shape (out_channels, in_channels, K_h, K_w).
    alpha : torch.nn.Parameter
        Learnable control parameter of shape (out_channels, in_channels) that interpolates
        between erosion and dilation.
    channel_merge_mode : str
        Channel merging strategy.
    padding_mode : str
        Border padding strategy.

    Notes
    -----
    The operation is defined as: Y(X) = σ(α) ⊕ X + (1 - σ(α)) ⊖ X
    where σ is the sigmoid function, ⊕ is dilation, and ⊖ is erosion.
    
    Both the structuring element and alpha parameter are trainable and optimized via backpropagation.
    Supports GPU acceleration via CUDA.

    Examples
    --------
    >>> import torch
    >>> from morphocore.nn import SigmaLayer
    >>> layer = SigmaLayer(in_channels=3, out_channels=16, kernel_size=5)
    >>> x = torch.randn(1, 3, 32, 32)
    >>> output = layer(x)
    >>> output.shape
    torch.Size([1, 16, 32, 32])
    
    Training example with alpha learning:
    
    >>> layer = SigmaLayer(3, 16, kernel_size=5, init_alpha=0.0)
    >>> optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)
    >>> x = torch.randn(8, 3, 32, 32)
    >>> output = layer(x)
    >>> loss = criterion(output, target)
    >>> loss.backward()
    >>> optimizer.step()
    >>> # Alpha will be learned to find optimal erosion/dilation balance

    See Also
    --------
    morphocore.functional.sigma_layer : Functional interface
    SMorph : Smooth morphological layer
    Dilation : Standard dilation layer
    Erosion : Standard erosion layer
    """

    def __init__(self, in_channels : int, out_channels : int, kernel_size : tuple, channel_merge_mode: str = "sum", padding_mode: str = "replicate", init_alpha: float = 0.0, dtype: torch.dtype = torch.float32, save_indices: bool = True, padding_value: float = 0.0, weight_dropout: float = 0.0, noise_scale: float = 0.0):
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        super().__init__(in_channels, out_channels, kernel_size, channel_merge_mode, init_alpha, dtype=dtype)
        self.save_indices = save_indices
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        self.weight_dropout = weight_dropout
        self.noise_scale = noise_scale

    def forward(self, x):
        return sigma_layer(x, self.weight, self.alpha, self.channel_merge_mode, padding_mode=self.padding_mode,
                      save_indices=self.save_indices, padding_value=self.padding_value,
                      weight_dropout=self.weight_dropout, training=self.training, noise_scale=self.noise_scale)