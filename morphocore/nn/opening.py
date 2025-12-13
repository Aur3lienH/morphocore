import torch
import torch.nn as nn
from .dilation import Dilation
from .erosion import Erosion


class Opening(nn.Module):
    """
    Learnable morphological opening layer (erosion followed by dilation).
    
    Opening is useful for removing small objects and noise, separating touching objects,
    and smoothing object boundaries from the inside.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels. Note: Must equal intermediate channels since dilation uses
        identity channel merging.
    kernel_size : int or tuple of int
        Size of the structuring element. If int, a square kernel is used.
    channel_merge_mode : str, optional
        Strategy for merging multiple input channels in the erosion step. Options include:
        'sum', 'max', 'min', 'average' or 'identity'. Default is 'sum'.
        Note: The dilation step always uses 'identity' mode.
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
    erosion : Erosion
        Learnable erosion layer with structuring element of shape (out_channels, in_channels, K_h, K_w).
    dilation : Dilation
        Learnable dilation layer with structuring element of shape (out_channels, out_channels, K_h, K_w).

    Notes
    -----
    The opening operation is defined as: (x ○ w) = ((x ⊖ w_e) ⊕ w_d)
    where ⊖ is erosion and ⊕ is dilation.
    
    The dilation step uses 'identity' channel merge mode to ensure each eroded channel
    is dilated independently, maintaining the morphological property where each erosion
    is paired with its corresponding dilation.
    
    Both structuring elements are trainable and optimized via backpropagation.
    Supports GPU acceleration via CUDA.

    Examples
    --------
    >>> import torch
    >>> from morphocore.nn import Opening
    >>> layer = Opening(in_channels=3, out_channels=16, kernel_size=5)
    >>> x = torch.randn(1, 3, 32, 32)
    >>> output = layer(x)
    >>> output.shape
    torch.Size([1, 16, 32, 32])
    
    Training example:
    
    >>> layer = Opening(3, 16, kernel_size=5)
    >>> optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)
    >>> x = torch.randn(8, 3, 32, 32)
    >>> output = layer(x)
    >>> loss = criterion(output, target)
    >>> loss.backward()
    >>> optimizer.step()

    See Also
    --------
    morphocore.functional.opening : Functional interface
    Closing : Morphological closing layer
    Dilation : Morphological dilation layer
    Erosion : Morphological erosion layer
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size, channel_merge_mode: str = "sum", 
                 padding_mode: str = "replicate", dtype: torch.dtype = torch.float32, save_indices: bool = True):
        super(Opening, self).__init__()
        
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        
        # Erosion: in_channels -> out_channels
        self.erosion = Erosion(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            channel_merge_mode=channel_merge_mode,
            padding_mode=padding_mode,
            dtype=dtype,
            save_indices=save_indices
        )
        
        # Dilation: out_channels -> out_channels (identity merge mode)
        self.dilation = Dilation(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            channel_merge_mode='identity',  # Always identity for opening
            padding_mode=padding_mode,
            dtype=dtype,
            save_indices=save_indices
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the opening layer.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, out_channels, height, width).
        """
        eroded = self.erosion(x)
        return self.dilation(eroded)