import torch
import torch.nn as nn
from .dilation import Dilation
from .erosion import Erosion


class Closing(nn.Module):
    """
    Learnable morphological closing layer (dilation followed by erosion).
    
    Closing is useful for filling small holes and gaps in the foreground, smoothing object contours,
    and connecting nearby objects.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels. Note: Must equal intermediate channels since erosion uses
        identity channel merging.
    kernel_size : int or tuple of int
        Size of the structuring element. If int, a square kernel is used.
    channel_merge_mode : str, optional
        Strategy for merging multiple input channels in the dilation step. Options include:
        'sum', 'max', 'min', 'average' or 'identity'. Default is 'sum'.
        Note: The erosion step always uses 'identity' mode.
    padding_mode : str, optional
        Padding strategy to handle border pixels. Options include:
        'geodesic' or 'replicate'. Default is 'replicate'.
    dtype : torch.dtype, optional
        Data type for the weight parameters. Default is torch.float32.
    save_indices : bool, optional
        Whether to save the indices of extremum values for gradient computation.
        Default is True.

    Attributes
    ----------
    dilation : Dilation
        Learnable dilation layer with structuring element of shape (out_channels, in_channels, K_h, K_w).
    erosion : Erosion
        Learnable erosion layer with structuring element of shape (out_channels, out_channels, K_h, K_w).

    Notes
    -----
    The closing operation is defined as: (x • w) = ((x ⊕ w_d) ⊖ w_e)
    where ⊕ is dilation and ⊖ is erosion.
    
    The erosion step uses 'identity' channel merge mode to ensure each dilated channel
    is eroded independently, maintaining the morphological property where each dilation
    is paired with its corresponding erosion.
    
    Both structuring elements are trainable and optimized via backpropagation.
    Supports GPU acceleration via CUDA.

    Examples
    --------
    >>> import torch
    >>> from morphocore.nn import Closing
    >>> layer = Closing(in_channels=3, out_channels=16, kernel_size=5)
    >>> x = torch.randn(1, 3, 32, 32)
    >>> output = layer(x)
    >>> output.shape
    torch.Size([1, 16, 32, 32])
    
    Training example:
    
    >>> layer = Closing(3, 16, kernel_size=5)
    >>> optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)
    >>> x = torch.randn(8, 3, 32, 32)
    >>> output = layer(x)
    >>> loss = criterion(output, target)
    >>> loss.backward()
    >>> optimizer.step()

    See Also
    --------
    morphocore.functional.closing : Functional interface
    Opening : Morphological opening layer
    Dilation : Morphological dilation layer
    Erosion : Morphological erosion layer
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size, channel_merge_mode: str = "sum", 
                 padding_mode: str = "replicate", dtype: torch.dtype = torch.float32, save_indices: bool = True):
        super(Closing, self).__init__()
        
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        
        self.dilation = Dilation(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            channel_merge_mode=channel_merge_mode,
            padding_mode=padding_mode,
            dtype=dtype,
            save_indices=save_indices
        )
        
        self.erosion = Erosion(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            channel_merge_mode='identity',
            padding_mode=padding_mode,
            dtype=dtype,
            save_indices=save_indices
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the closing layer.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, out_channels, height, width).
        """
        dilated = self.dilation(x)
        return self.erosion(dilated)