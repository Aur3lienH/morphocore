import pytest
import torch

from morphocore.nn import Closing, Opening
from .conftest import device


LAYERS = [(Opening, 'anti'), (Closing, 'ext')]


def trained_like(layer):
    """Move the structuring element off its zero initialisation, as training would."""
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.randn_like(p) * 0.5)
    return layer


def violation_pct(y, x, sense):
    if sense == 'anti':
        return (y > x + 1e-5).float().mean().item() * 100
    return (y < x - 1e-5).float().mean().item() * 100


@pytest.mark.parametrize('cls,sense', LAYERS)
class TestSharedStructuringElement:

    def test_shared_by_default(self, cls, sense, device):
        layer = cls(3, 3, 5).to(device)
        first, second = layer.erosion, layer.dilation
        assert (first.weight is second.weight)
        assert len(list(layer.parameters())) == 1
        assert tuple(layer.erosion.weight.shape) == (3, 1, 5, 5)

    def test_opt_out_gives_two_parameters(self, cls, sense, device):
        layer = cls(3, 3, 5, share_weights=False).to(device)
        assert layer.erosion.weight is not layer.dilation.weight
        assert len(list(layer.parameters())) == 2

    def test_shared_is_idempotent_and_ordered(self, cls, sense, device):
        """With one structuring element and geodesic padding this is an algebraic operator."""
        layer = trained_like(cls(3, 3, 5, padding_mode='geodesic').to(device))
        x = torch.randn(2, 3, 20, 20, device=device)

        y = layer(x)
        assert violation_pct(y, x, sense) == 0.0
        assert (layer(y) - y).abs().max().item() < 1e-5

    def test_independent_weights_lose_the_property(self, cls, sense, device):
        """The opt-out is more expressive but is not an opening/closing any more."""
        layer = trained_like(cls(3, 3, 5, padding_mode='geodesic', share_weights=False).to(device))
        x = torch.randn(2, 3, 20, 20, device=device)

        y = layer(x)
        assert (layer(y) - y).abs().max().item() > 1e-3

    def test_interior_holds_for_every_padding_mode(self, cls, sense, device):
        """Padding can only perturb the border, never the interior."""
        x = torch.randn(2, 3, 24, 24, device=device)
        for padding_mode in ('zero', 'geodesic', 'constant', 'replicate'):
            layer = cls(3, 3, 5, padding_mode=padding_mode).to(device)
            trained_like(layer)
            y = layer(x)
            inner = (slice(None), slice(None), slice(4, -4), slice(4, -4))
            assert violation_pct(y[inner], x[inner], sense) == 0.0, padding_mode

    def test_gradient_reaches_the_shared_weight(self, cls, sense, device):
        layer = trained_like(cls(3, 3, 5).to(device))
        layer(torch.randn(2, 3, 16, 16, device=device)).sum().backward()

        grad = layer.erosion.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0

    def test_sharing_rejects_channel_mismatch(self, cls, sense, device):
        with pytest.raises(ValueError, match='in_channels == out_channels'):
            cls(3, 8, 5)

    def test_sharing_rejects_channel_merging(self, cls, sense, device):
        with pytest.raises(ValueError, match="channel_merge_mode='identity'"):
            cls(3, 3, 5, channel_merge_mode='sum')

    def test_opt_out_still_allows_channel_mismatch(self, cls, sense, device):
        layer = cls(3, 8, 5, share_weights=False).to(device)
        assert layer(torch.randn(2, 3, 16, 16, device=device)).shape == (2, 8, 16, 16)
