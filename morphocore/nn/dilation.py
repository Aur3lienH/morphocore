from .morpho_module import MorphoModule
from morphocore.functional import dilation
import torch


class Dilation(MorphoModule):
    """
    Learnable morphological dilation layer.

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
        'geodesic' or 'replicate'. Default is 'geodesic'.
    dtype : torch.dtype, optional
        Data type for the weight parameter. Default is torch.float32.
    save_indices : bool, optional
        Whether to save the indices of maximum values for gradient computation.
        Default is True.

    Attributes
    ----------
    weight : torch.nn.Parameter
        Learnable structuring element of shape (out_channels, in_channels, K_h, K_w).
    channel_merge_mode : str
        Channel merging strategy.
    padding_mode : str
        Border padding strategy.

    Notes
    -----
    This layer performs morphological dilation with learnable weights, defined as:
    (x ⊕ w)[i,j] = max{x[i+m, j+n] + w[m,n]}
    
    The structuring element is trainable and can be optimized via backpropagation.
    Supports GPU acceleration via CUDA.

    Examples
    --------
    >>> import torch
    >>> from morphocore.nn import Dilation
    >>> layer = Dilation(in_channels=3, out_channels=16, kernel_size=5)
    >>> x = torch.randn(1, 3, 32, 32)
    >>> output = layer(x)
    >>> output.shape
    torch.Size([1, 16, 32, 32])
    
    Training example:
    
    >>> layer = Dilation(3, 16, kernel_size=5)
    >>> optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)
    >>> x = torch.randn(8, 3, 32, 32)
    >>> output = layer(x)
    >>> loss = criterion(output, target)
    >>> loss.backward()
    >>> optimizer.step()

    See Also
    --------
    morphocore.functional.dilation : Functional interface
    Erosion : Morphological erosion layer
    """
    def __init__(self, in_channels : int, out_channels : int, kernel_size : tuple, channel_merge_mode: str = "sum", padding_mode: str = "geodesic", dtype: torch.dtype = torch.float32, save_indices: bool = True):
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        self.save_indices = save_indices
        super().__init__(in_channels, out_channels, kernel_size, channel_merge_mode, padding_mode, dtype=dtype)

    def forward(self, x):
        return dilation(x, self.weight, self.channel_merge_mode, self.padding_mode, self.save_indices)