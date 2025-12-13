import torch
import torch.nn as nn
from .alpha_module import AlphaModule
from morphocore.functional import smorph


class SMorph(AlphaModule):
    """
    Learnable smooth morphological layer with approximated interpolation between erosion and dilation.
    
    This layer uses a smooth approximation to transition from erosion to dilation based on a learnable 
    alpha parameter. When alpha → +∞, the operation becomes dilation; when alpha → -∞, it becomes erosion.

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
    init_alpha : float, optional
        Initial value for the alpha parameter. Default is 0.0.
    dtype : torch.dtype, optional
        Data type for the weight and alpha parameters. Default is torch.float32.
    init_beta : float, optional
        Initial value for the beta parameter for selective channel merging. 
        If None, beta is not used. Default is None.

    Attributes
    ----------
    weight : torch.nn.Parameter
        Learnable structuring element of shape (out_channels, in_channels, K_h, K_w).
    alpha : torch.nn.Parameter
        Learnable control parameter of shape (out_channels, in_channels) that interpolates
        between erosion and dilation.
    beta : torch.nn.Parameter or None
        Optional learnable parameter for selective channel merging of shape (out_channels, in_channels).
    channel_merge_mode : str
        Channel merging strategy.

    Notes
    -----
    The operation uses a smooth approximation with tanh activation:
    w̃(α) = tanh(α) × [(1-tanh(α))/2 × w_flipped + (1+tanh(α))/2 × w]
    
    This provides a differentiable approximation of the morphological interpolation, making it
    more suitable for gradient-based optimization than exact morphological operations.
    
    Both the structuring element and alpha parameter are trainable and optimized via backpropagation.
    Supports GPU acceleration via CUDA.

    Examples
    --------
    >>> import torch
    >>> from morphocore.nn import SMorph
    >>> layer = SMorph(in_channels=3, out_channels=16, kernel_size=5)
    >>> x = torch.randn(1, 3, 32, 32)
    >>> output = layer(x)
    >>> output.shape
    torch.Size([1, 16, 32, 32])
    
    Training example with alpha learning:
    
    >>> layer = SMorph(3, 16, kernel_size=5, init_alpha=0.0)
    >>> optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)
    >>> x = torch.randn(8, 3, 32, 32)
    >>> output = layer(x)
    >>> loss = criterion(output, target)
    >>> loss.backward()
    >>> optimizer.step()
    
    Using beta parameter for selective channel merging:
    
    >>> layer = SMorph(3, 16, kernel_size=5, init_beta=1.0)
    >>> output = layer(x)

    See Also
    --------
    morphocore.functional.smorph : Functional interface
    EMorph : Exact morphological interpolation layer
    Dilation : Standard dilation layer
    Erosion : Standard erosion layer
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, channel_merge_mode: str = "sum", init_alpha: float = 0.0, dtype: torch.dtype = torch.float32, init_beta: float = None):
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        super(SMorph, self).__init__(in_channels, out_channels, kernel_size, channel_merge_mode, init_alpha, dtype)
        if init_beta is not None:
            self.beta = nn.Parameter(torch.full((out_channels, in_channels), init_beta, dtype=dtype))
        else:
            self.beta = None

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the smooth morphological layer.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, out_channels, height, width).
        """
        return smorph(x=x, w=self.weight, alpha=self.alpha, beta=self.beta, channel_merge_mode=self.channel_merge_mode)