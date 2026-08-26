import numpy as np
import pytest
import torch

import morphocore.functional as F
from .conftest import device


def reference_morphology(image, selem, operation, padding_mode, padding_value=0.0):
    """
    Numpy reference for a single channel morphological operation with an explicit padding mode.

    Mirrors utils.correct_morphology, the only difference being how the border is filled, so any
    divergence between the two is a padding bug and not a convention mismatch.
    """
    pad_h, pad_w = selem.shape[0] // 2, selem.shape[1] // 2

    if operation == 'dilation':
        op_func, result = np.maximum, np.full_like(image, -np.inf)
        geodesic_fill = -np.inf
    elif operation == 'erosion':
        op_func, result = np.minimum, np.full_like(image, np.inf)
        geodesic_fill = np.inf
        selem = np.flip(selem, axis=(-2, -1))
    else:
        raise ValueError(f"Unknown operation: {operation}")

    if padding_mode == 'replicate':
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    else:
        fill = {'zero': 0.0, 'geodesic': geodesic_fill, 'constant': padding_value}[padding_mode]
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=fill)

    for dy in range(selem.shape[0]):
        for dx in range(selem.shape[1]):
            y_start = pad_h - (dy - pad_h)
            x_start = pad_w - (dx - pad_w)
            shifted = padded[y_start:y_start + image.shape[0], x_start:x_start + image.shape[1]]

            if operation == 'dilation':
                result = op_func(result, shifted + selem[dy, dx])
            else:
                result = op_func(result, shifted - selem[dy, dx])

    return result


def apply_operation(operation, image, selem, padding_mode, padding_value=0.0, channel_merge_mode='max'):
    fn = F.dilation if operation == 'dilation' else F.erosion
    return fn(image, selem, channel_merge_mode=channel_merge_mode, padding_mode=padding_mode,
              padding_value=padding_value)


@pytest.mark.parametrize('operation', ['dilation', 'erosion'])
@pytest.mark.parametrize('padding_mode,padding_value', [
    ('zero', 0.0),
    ('geodesic', 0.0),
    ('constant', 0.0),
    ('constant', -3.5),
    ('constant', 7.25),
    ('replicate', 0.0),
])
@pytest.mark.parametrize('kernel_size', [3, 5])
class TestPaddingAgainstReference:

    def test_matches_reference(self, device, operation, padding_mode, padding_value, kernel_size):
        rng = np.random.default_rng(0)
        image_np = rng.standard_normal((17, 23))
        kernel_np = rng.standard_normal((kernel_size, kernel_size))

        expected = reference_morphology(image_np, kernel_np, operation, padding_mode, padding_value)

        image = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0).to(device)
        kernel = torch.from_numpy(kernel_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        result = apply_operation(operation, image, kernel, padding_mode, padding_value)

        np.testing.assert_allclose(result.squeeze().cpu().numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_matches_reference_double(self, device, operation, padding_mode, padding_value, kernel_size):
        rng = np.random.default_rng(1)
        image_np = rng.standard_normal((11, 9))
        kernel_np = rng.standard_normal((kernel_size, kernel_size))

        expected = reference_morphology(image_np, kernel_np, operation, padding_mode, padding_value)

        image = torch.from_numpy(image_np).double().unsqueeze(0).unsqueeze(0).to(device)
        kernel = torch.from_numpy(kernel_np).unsqueeze(0).unsqueeze(0).to(device)

        result = apply_operation(operation, image, kernel, padding_mode, padding_value)

        np.testing.assert_allclose(result.squeeze().cpu().numpy(), expected, rtol=1e-10, atol=1e-10)


class TestPaddingEquivalences:
    """The value based modes are one implementation, so their documented aliases must agree exactly."""

    @pytest.mark.parametrize('operation', ['dilation', 'erosion'])
    def test_zero_equals_constant_zero(self, device, operation):
        image = torch.randn(2, 3, 16, 16, device=device)
        kernel = torch.randn(4, 3, 3, 3, device=device)

        zero = apply_operation(operation, image, kernel, 'zero')
        constant = apply_operation(operation, image, kernel, 'constant', 0.0)

        torch.testing.assert_close(zero, constant)

    @pytest.mark.parametrize('operation', ['dilation', 'erosion'])
    def test_geodesic_never_wins(self, device, operation):
        """A geodesic border must leave the result identical to a constant border far past any value."""
        image = torch.randn(1, 1, 12, 12, device=device)
        kernel = torch.zeros(1, 1, 5, 5, device=device)

        unreachable = -1e9 if operation == 'dilation' else 1e9
        geodesic = apply_operation(operation, image, kernel, 'geodesic')
        constant = apply_operation(operation, image, kernel, 'constant', unreachable)

        torch.testing.assert_close(geodesic, constant)

    @pytest.mark.parametrize('operation', ['dilation', 'erosion'])
    def test_constant_border_dominates(self, device, operation):
        """A constant chosen past the data must take over every output touching the border."""
        image = torch.rand(1, 1, 8, 8, device=device)
        kernel = torch.zeros(1, 1, 3, 3, device=device)
        extreme = 50.0 if operation == 'dilation' else -50.0

        result = apply_operation(operation, image, kernel, 'constant', extreme).squeeze()

        border = torch.ones_like(result, dtype=torch.bool)
        border[1:-1, 1:-1] = False
        assert torch.all(result[border] == extreme)
        assert torch.all(result[~border] != extreme)

    @pytest.mark.parametrize('padding_mode', ['zero', 'geodesic', 'constant', 'replicate'])
    def test_interior_is_padding_independent(self, device, padding_mode):
        """Padding may only ever change pixels within half a kernel of the border."""
        image = torch.randn(1, 1, 20, 20, device=device)
        kernel = torch.randn(1, 1, 5, 5, device=device)

        reference = apply_operation('dilation', image, kernel, 'geodesic').squeeze()
        result = apply_operation('dilation', image, kernel, padding_mode, 4.0).squeeze()

        torch.testing.assert_close(result[2:-2, 2:-2], reference[2:-2, 2:-2])


class TestPaddingModesPlumbing:

    @pytest.mark.parametrize('padding_mode', ['zero', 'geodesic', 'constant', 'replicate'])
    @pytest.mark.parametrize('channel_merge_mode', ['sum', 'max', 'min', 'mean', 'identity'])
    def test_every_merge_mode_accepts_every_padding(self, device, padding_mode, channel_merge_mode):
        channels = 3
        image = torch.randn(2, channels, 12, 12, device=device)
        weight_in = 1 if channel_merge_mode == 'identity' else channels
        kernel = torch.randn(channels, weight_in, 3, 3, device=device)

        result = apply_operation('dilation', image, kernel, padding_mode, 1.5,
                                 channel_merge_mode=channel_merge_mode)

        assert result.shape == (2, channels, 12, 12)
        assert torch.isfinite(result).all()

    @pytest.mark.parametrize('padding_mode', ['zero', 'geodesic', 'constant', 'replicate'])
    @pytest.mark.parametrize('operation', ['dilation', 'erosion'])
    def test_backward_runs_and_reaches_the_weight(self, device, padding_mode, operation):
        image = torch.randn(1, 2, 10, 10, device=device, requires_grad=True)
        kernel = torch.randn(2, 2, 3, 3, device=device, requires_grad=True)

        apply_operation(operation, image, kernel, padding_mode, 0.5).sum().backward()

        assert image.grad is not None and torch.isfinite(image.grad).all()
        assert kernel.grad is not None and torch.isfinite(kernel.grad).all()
        assert kernel.grad.abs().sum() > 0

    @pytest.mark.parametrize('padding_mode', ['zero', 'geodesic', 'constant', 'replicate'])
    def test_kernel_larger_than_image(self, device, padding_mode):
        """Exercises the gather fallback the CUDA replicate path needs when the pad exceeds the image."""
        image = torch.randn(1, 1, 3, 3, device=device)
        kernel = torch.zeros(1, 1, 9, 9, device=device)

        result = apply_operation('dilation', image, kernel, padding_mode, 2.0)

        assert result.shape == (1, 1, 3, 3)
        assert torch.isfinite(result).all()

    def test_unknown_padding_mode_is_rejected(self):
        image = torch.randn(1, 1, 8, 8)
        kernel = torch.zeros(1, 1, 3, 3)

        with pytest.raises(ValueError, match='Unknown padding mode'):
            F.dilation(image, kernel, padding_mode='reflect')

    def test_padding_value_defaults_to_zero(self, device):
        image = torch.randn(1, 1, 8, 8, device=device)
        kernel = torch.zeros(1, 1, 3, 3, device=device)

        torch.testing.assert_close(
            F.dilation(image, kernel, padding_mode='constant'),
            F.dilation(image, kernel, padding_mode='zero'),
        )


class TestPaddingCudaMatchesCpu:

    @pytest.mark.parametrize('padding_mode', ['zero', 'geodesic', 'constant', 'replicate'])
    @pytest.mark.parametrize('operation', ['dilation', 'erosion'])
    @pytest.mark.parametrize('channel_merge_mode', ['sum', 'max', 'min', 'mean'])
    def test_forward_parity(self, padding_mode, operation, channel_merge_mode):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(7)
        image = torch.randn(2, 3, 24, 24)
        kernel = torch.randn(4, 3, 5, 5)

        cpu = apply_operation(operation, image, kernel, padding_mode, -2.5,
                              channel_merge_mode=channel_merge_mode)
        cuda = apply_operation(operation, image.cuda(), kernel.cuda(), padding_mode, -2.5,
                               channel_merge_mode=channel_merge_mode)

        torch.testing.assert_close(cuda.cpu(), cpu, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize('padding_mode', ['zero', 'geodesic', 'constant', 'replicate'])
    @pytest.mark.parametrize('operation', ['dilation', 'erosion'])
    def test_backward_parity(self, padding_mode, operation):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(11)
        image = torch.randn(1, 2, 16, 16)
        kernel = torch.randn(2, 2, 3, 3)

        grads = []
        for dev in ('cpu', 'cuda'):
            x = image.clone().to(dev).requires_grad_(True)
            w = kernel.clone().to(dev).requires_grad_(True)
            apply_operation(operation, x, w, padding_mode, -2.5).sum().backward()
            grads.append((x.grad.cpu(), w.grad.cpu()))

        torch.testing.assert_close(grads[1][0], grads[0][0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(grads[1][1], grads[0][1], rtol=1e-4, atol=1e-4)
