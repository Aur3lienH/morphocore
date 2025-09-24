import pytest
import torch
import numpy as np
import morphocore.functional as F



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
        
        if operation == 'dilation':
            result = benchmark(F.dilation, image_torch, selem_torch)
        else:
            result = benchmark(F.erosion, image_torch, selem_torch)
        
        assert result.shape == (batch_size, 1, image_size, image_size)


    @pytest.mark.parametrize("image_size", [32, 64, 128])
    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    @pytest.mark.parametrize("batch_size", [1, 16, 32, 64])
    def test_benchmark_backward(self, benchmark, device, operation, image_size, kernel_size, batch_size):
        images = [np.random.randn(image_size, image_size) for _ in range(batch_size)]
        image_np = np.stack(images)
        
        kernel = np.random.randn(kernel_size, kernel_size)
        
        # Create tensors with gradients enabled
        image_torch = torch.from_numpy(image_np).float().unsqueeze(1).to(device)
        image_torch.requires_grad_(True)
        
        selem_torch = torch.from_numpy(kernel.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        selem_torch.requires_grad_(True)
        
        # Forward pass (not benchmarked)
        if operation == 'dilation':
            output = F.dilation(image_torch, selem_torch)
        else:
            output = F.erosion(image_torch, selem_torch)
        
        # Create gradient output (simulates loss.backward())
        grad_output = torch.randn_like(output)
        
        def backward_pass():
            # Clear any existing gradients
            if image_torch.grad is not None:
                image_torch.grad.zero_()
            if selem_torch.grad is not None:
                selem_torch.grad.zero_()
            
            # Perform backward pass
            torch.autograd.backward(output, grad_output, retain_graph=True)
            
            return image_torch.grad, selem_torch.grad
        
        # Benchmark only the backward pass
        grad_input, grad_kernel = benchmark(backward_pass)
        
        # Verify gradient shapes
        assert grad_input.shape == image_torch.shape
        assert grad_kernel.shape == selem_torch.shape