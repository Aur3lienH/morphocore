import pytest
import torch
import numpy as np
import morphocore.functional as F
from .utils import correct_morphology, dispatch_operation
from .conftest import device, operation


class TestMorphologyBenchmark:
    @pytest.mark.parametrize("image_size", [32, 64, 128])
    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    @pytest.mark.parametrize("batch_size", [1, 16, 32, 64])
    def test_benchmark(self, benchmark, device, operation, image_size, kernel_size, batch_size):
        images = [np.random.randn(image_size, image_size) for _ in range(batch_size)]
        image_np = np.stack(images)
        
        kernel = np.random.randn(kernel_size, kernel_size)
        
        image_torch = torch.from_numpy(image_np).float().unsqueeze(1).to(device)
        selem_torch = torch.from_numpy(kernel.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        
        def run_operation():
            return dispatch_operation(operation, image_torch, selem_torch)
        
        result = benchmark(run_operation)
        
        assert result.shape == (batch_size, 1, image_size, image_size)


    @pytest.mark.parametrize("image_size", [32, 64, 128])
    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    @pytest.mark.parametrize("batch_size", [1, 16, 32, 64])
    def test_benchmark_backward(self, benchmark, device, operation, image_size, kernel_size, batch_size):
        images = [np.random.randn(image_size, image_size) for _ in range(batch_size)]
        image_np = np.stack(images)
        
        kernel = np.random.randn(kernel_size, kernel_size)
        
        image_torch = torch.from_numpy(image_np).float().unsqueeze(1).to(device)
        image_torch.requires_grad_(True)
        
        selem_torch = torch.from_numpy(kernel.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        selem_torch.requires_grad_(True)
        
        output = dispatch_operation(operation, image_torch, selem_torch)
        
        grad_output = torch.randn_like(output)
        
        def backward_pass():
            if image_torch.grad is not None:
                image_torch.grad.zero_()
            if selem_torch.grad is not None:
                selem_torch.grad.zero_()
            
            torch.autograd.backward(output, grad_output, retain_graph=True)
            
            return image_torch.grad, selem_torch.grad
        
        grad_input, grad_kernel = benchmark(backward_pass)
        
        assert grad_input.shape == image_torch.shape
        assert grad_kernel.shape == selem_torch.shape