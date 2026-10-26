import pytest
import torch

import morphocore.functional as F
import morphocore.nn as N


MERGES = ['sum', 'max', 'min', 'mean', 'identity']


def make_inputs(merge, dtype=torch.float32, device='cpu'):
    torch.manual_seed(0)
    x = torch.randn(3, 4, 9, 11, dtype=dtype).to(device)
    w = torch.randn(4, 1 if merge == 'identity' else 4, 3, 5, dtype=dtype).to(device)
    return x, w


def draw_noise(w, batch_size, noise_scale):
    """Same draw as the extension: one weight shaped noise per batch element, in [-s, s], on the
    weight's device."""
    return (torch.rand((batch_size,) + tuple(w.shape), dtype=w.dtype, device=w.device) * 2.0 - 1.0) * noise_scale


def per_sample_reference(op, x, w, noise, merge, save_indices):
    """Run each batch element alone with its own noisy weight, without noise in the kernel."""
    outputs = []
    for b in range(x.shape[0]):
        if op == 'dilation':
            outputs.append(F.dilation(x[b:b + 1], w + noise[b], merge, save_indices=save_indices))
        else:
            # The extension adds the noise to the flipped weight of the erosion.
            noisy = (w.flip((-2, -1)) + noise[b]).flip((-2, -1))
            outputs.append(F.erosion(x[b:b + 1], noisy, merge, save_indices=save_indices))
    return torch.cat(outputs)


@pytest.mark.parametrize('save_indices', [True, False])
@pytest.mark.parametrize('merge', MERGES)
@pytest.mark.parametrize('op', ['dilation', 'erosion'])
class TestWeightNoise:

    def test_matches_per_sample_reference(self, op, merge, save_indices, device):
        x, w = make_inputs(merge, device=device)
        x.requires_grad_(True)
        w.requires_grad_(True)
        fn = F.dilation if op == 'dilation' else F.erosion

        torch.manual_seed(7)
        out = fn(x, w, merge, save_indices=save_indices, training=True, noise_scale=0.3)
        grad = torch.randn_like(out)
        grad_x, grad_w = torch.autograd.grad(out, (x, w), grad)

        torch.manual_seed(7)
        noise = draw_noise(w.detach(), x.shape[0], 0.3)
        expected = per_sample_reference(op, x, w, noise, merge, save_indices)
        expected_grad_x, expected_grad_w = torch.autograd.grad(expected, (x, w), grad)

        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(grad_x, expected_grad_x)
        torch.testing.assert_close(grad_w, expected_grad_w)

    def test_gradcheck(self, op, merge, save_indices, device):
        x, w = make_inputs(merge, torch.float64, device)
        x = x[:2, :, :5, :6].clone().requires_grad_(True)
        w = w[..., :3].clone().requires_grad_(True)
        fn = F.dilation if op == 'dilation' else F.erosion

        def noisy(x, w):
            # The same noise on every evaluation, so the finite differences see a fixed function.
            # Geodesic padding: replicate does not route the gradient to the copied edge pixel.
            torch.manual_seed(3)
            return fn(x, w, merge, 'geodesic', save_indices, training=True, noise_scale=0.5)

        assert torch.autograd.gradcheck(noisy, (x, w))


@pytest.mark.parametrize('op', ['dilation', 'erosion'])
def test_noise_varies_per_sample(op):
    x, w = make_inputs('sum')
    # Identical batch elements only differ through their noise.
    x = x[:1].expand(4, -1, -1, -1).contiguous()
    fn = F.dilation if op == 'dilation' else F.erosion

    out = fn(x, w, training=True, noise_scale=0.5)

    assert not torch.equal(out[0], out[1])
    noise_free = fn(x[:1], w)
    # Every weight moves by at most 0.5, and so does each extremum; the sum merges 4 channels.
    assert (out - noise_free).abs().max() <= 4 * 0.5 + 1e-5


@pytest.mark.parametrize('op', ['dilation', 'erosion'])
def test_inactive_outside_training(op):
    x, w = make_inputs('sum')
    fn = F.dilation if op == 'dilation' else F.erosion

    torch.testing.assert_close(fn(x, w, training=False, noise_scale=0.5), fn(x, w))


def test_rejects_negative_scale():
    x, w = make_inputs('sum')
    with pytest.raises(RuntimeError, match="noise_scale"):
        F.dilation(x, w, training=True, noise_scale=-0.1)


@pytest.mark.parametrize('layer', [
    lambda: N.Dilation(3, 3, 3, noise_scale=0.5),
    lambda: N.Erosion(3, 3, 3, noise_scale=0.5),
    lambda: N.Opening(3, 3, 3, noise_scale=0.5),
    lambda: N.Closing(3, 3, 3, noise_scale=0.5),
    lambda: N.SigmaLayer(3, 3, 3, noise_scale=0.5),
], ids=['Dilation', 'Erosion', 'Opening', 'Closing', 'SigmaLayer'])
def test_layers_follow_training_mode(layer):
    torch.manual_seed(0)
    module = layer()
    x = torch.randn(2, 3, 12, 12)

    module.eval()
    torch.testing.assert_close(module(x), module(x))

    module.train()
    torch.manual_seed(1)
    first = module(x)
    torch.manual_seed(2)
    second = module(x)
    assert not torch.equal(first, second)
    first.sum().backward()
