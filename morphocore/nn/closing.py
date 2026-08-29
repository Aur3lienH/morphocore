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
        Only meaningful when share_weights is False, where it defaults to 'sum'.
        Strategy for merging multiple input channels in the dilation step. Options include:
        'sum', 'max', 'min', 'average' or 'identity'. Default is 'sum'.
        Note: The erosion step always uses 'identity' mode.
    padding_mode : str, optional
        Padding strategy to handle border pixels. Options include:
        'zero', 'geodesic', 'constant' or 'replicate'. Default is 'replicate'.
    padding_value : float, optional
        Border value used when padding_mode is 'constant'. Ignored by the other
        padding modes. Default is 0.0.
    dtype : torch.dtype, optional
        Data type for the weight parameters. Default is torch.float32.
    save_indices : bool, optional
        Whether to save the indices of extremum values for gradient computation.
        Default is True.
    weight_dropout : float, optional
        Probability of dropping weights in the structuring element during training.
        Each stage draws its own mask. Must be in range [0, 1). Default is 0.0 (no dropout).
    share_weights : bool, optional
        Whether both stages use a single structuring element. Default is True, which
        is what makes this layer a genuine morphological closing. Requires
        in_channels == out_channels and forces channel_merge_mode='identity', so that
        each channel is closined on its own. Set to False for two independent
        structuring elements, which is more expressive but no longer a closing.

    Attributes
    ----------
    dilation : Dilation
        Learnable dilation layer with structuring element of shape (out_channels, in_channels, K_h, K_w).
    erosion : Erosion
        Learnable erosion layer with structuring element of shape (out_channels, 1, K_h, K_w),
        one plane per channel because this stage uses identity channel merging.

    Notes
    -----
    With share_weights=True (the default) both stages use the same structuring element w,
    giving the algebraic closing: extensive (φ(x) ≥ x) and idempotent.
    Both properties hold exactly in the interior; at the border they depend on padding_mode,
    and only 'geodesic' preserves them everywhere.

    With share_weights=False the two stages learn independent structuring elements w_e and
    w_d. That is strictly more expressive but neither extensive nor idempotent.
    Stage order is dilation then erosion; the second stage always merges with
    'identity' so each channel of the first stage is paired with its own.

    The structuring element(s) are trainable and optimized via backpropagation.
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

    def __init__(self, in_channels: int, out_channels: int, kernel_size, channel_merge_mode: str = None,
                 padding_mode: str = "replicate", dtype: torch.dtype = torch.float32, save_indices: bool = True,
                 padding_value: float = 0.0, share_weights: bool = True,
                 weight_dropout: float = 0.0):
        super(Closing, self).__init__()

        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)

        # Sharing one structuring element between the two stages is what makes this an algebraic
        # closing: anti-extensive (resp. extensive) and idempotent. It only makes sense per channel,
        # so both stages merge with 'identity' and the two channel counts have to agree.
        if share_weights:
            if in_channels != out_channels:
                raise ValueError(
                    f"share_weights=True needs in_channels == out_channels, got {in_channels} and "
                    f"{out_channels}. Pass share_weights=False for two independent structuring elements."
                )
            if channel_merge_mode not in (None, 'identity'):
                raise ValueError(
                    f"share_weights=True forces channel_merge_mode='identity', got "
                    f"'{channel_merge_mode}'. Pass share_weights=False to merge channels instead."
                )
            channel_merge_mode = 'identity'
        elif channel_merge_mode is None:
            channel_merge_mode = 'sum'

        self.share_weights = share_weights
        
        self.dilation = Dilation(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            channel_merge_mode=channel_merge_mode,
            padding_mode=padding_mode,
            padding_value=padding_value,
            weight_dropout=weight_dropout,
            dtype=dtype,
            save_indices=save_indices
        )
        
        self.erosion = Erosion(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            channel_merge_mode='identity',
            padding_mode=padding_mode,
            padding_value=padding_value,
            weight_dropout=weight_dropout,
            dtype=dtype,
            save_indices=save_indices
        )

        if share_weights:
            # One nn.Parameter behind both stages; torch de-duplicates it in parameters().
            self.erosion.weight = self.dilation.weight

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