import pytest
import torch

import morphocore.functional as F
import morphocore.nn as N
from .conftest import device


def randomised(layer):
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.randn_like(p) * 0.5)
    return layer


def make_weights(dev):
    """Fixed structuring elements, so only the dropout draw can vary between calls."""
    torch.manual_seed(1234)
    return (torch.randn(3, 3, 3, 3, device=dev),
            torch.randn(3, 1, 3, 3, device=dev),
            torch.randn(3, 3, device=dev))


def functional_call(name, x, weights, dropout, training):
    w, w_second, alpha = weights
    common = dict(weight_dropout=dropout, training=training)
    if name == 'opening':
        return F.opening(x, w, w_second, **common)
    if name == 'closing':
        return F.closing(x, w, w_second, **common)
    return F.emorph(x, w, alpha, **common)


@pytest.mark.parametrize('name', ['opening', 'closing', 'emorph'])
class TestFunctionalWeightDropout:

    def test_inactive_outside_training(self, name, device):
        torch.manual_seed(0)
        x = torch.randn(1, 3, 16, 16, device=device)
        weights = make_weights(device)

        torch.manual_seed(1)
        first = functional_call(name, x, weights, 0.5, training=False)
        torch.manual_seed(2)
        second = functional_call(name, x, weights, 0.5, training=False)

        torch.testing.assert_close(first, second)

    def test_active_during_training(self, name, device):
        torch.manual_seed(0)
        x = torch.randn(1, 3, 16, 16, device=device)

        weights = make_weights(device)

        torch.manual_seed(1)
        dropped = functional_call(name, x, weights, 0.5, training=True)
        torch.manual_seed(1)
        plain = functional_call(name, x, weights, 0.0, training=True)

        assert not torch.equal(dropped, plain)

    def test_zero_probability_is_a_no_op(self, name, device):
        torch.manual_seed(0)
        x = torch.randn(1, 3, 16, 16, device=device)

        weights = make_weights(device)

        torch.manual_seed(1)
        training = functional_call(name, x, weights, 0.0, training=True)
        torch.manual_seed(1)
        inference = functional_call(name, x, weights, 0.0, training=False)

        torch.testing.assert_close(training, inference)


@pytest.mark.parametrize('cls', [N.Opening, N.Closing, N.EMorph])
class TestModuleWeightDropout:

    def test_follows_train_and_eval(self, cls, device):
        layer = randomised(cls(3, 3, 3, weight_dropout=0.5).to(device))
        x = torch.randn(2, 3, 16, 16, device=device)

        layer.eval()
        torch.testing.assert_close(layer(x), layer(x))

        layer.train()
        assert not torch.equal(layer(x), layer(x))

    def test_defaults_to_disabled(self, cls, device):
        layer = randomised(cls(3, 3, 3).to(device)).train()
        x = torch.randn(2, 3, 16, 16, device=device)

        assert layer.weight_dropout == 0.0 if hasattr(layer, 'weight_dropout') else True
        torch.testing.assert_close(layer(x), layer(x))

    def test_gradients_survive_dropout(self, cls, device):
        # 'zero' rather than the default: see TestDropoutGeodesicInteraction below.
        layer = randomised(cls(3, 3, 3, weight_dropout=0.3, padding_mode='zero').to(device)).train()

        layer(torch.randn(2, 3, 16, 16, device=device)).sum().backward()

        for name, p in layer.named_parameters():
            assert p.grad is not None, name
            assert torch.isfinite(p.grad).all(), name


class TestDropoutGeodesicInteraction:
    """
    Dropped weights are set to -inf and geodesic padding fills the border with -inf too, so a
    border pixel whose only surviving tap falls outside the image has no finite candidate and the
    operation returns +/-inf. Predates weight dropout being wired into opening/closing/emorph;
    those merely surface it in a gradient rather than only in the output.
    """

    def test_dilation_border_goes_infinite(self, device):
        x = torch.randn(1, 1, 8, 8, device=device)
        w = torch.full((1, 1, 3, 3), -float('inf'), device=device)
        w[0, 0, 0, 0] = 0.0  # a single surviving tap, as heavy dropout would leave

        assert torch.isinf(F.dilation(x, w, padding_mode='geodesic')).any()
        assert not torch.isinf(F.dilation(x, w, padding_mode='zero')).any()
        assert not torch.isinf(F.dilation(x, w, padding_mode='replicate')).any()

    def test_emorph_gradient_goes_infinite(self, device):
        """
        Deterministic stand-in for a heavy dropout draw. Relying on an actual dropout draw would
        be flaky, since only some draws leave a border pixel with no finite candidate.
        """
        w = torch.full((1, 1, 3, 3), -float('inf'), device=device)
        w[0, 0, 0, 0] = 0.0
        x = torch.randn(1, 1, 8, 8, device=device)

        alpha = torch.zeros(1, 1, device=device, requires_grad=True)
        F.emorph(x, w, alpha, padding_mode='geodesic').sum().backward()
        assert not torch.isfinite(alpha.grad).all(), "known defect appears to be fixed, drop this test"

        safe = torch.zeros(1, 1, device=device, requires_grad=True)
        F.emorph(x, w, safe, padding_mode='zero').sum().backward()
        assert torch.isfinite(safe.grad).all()
