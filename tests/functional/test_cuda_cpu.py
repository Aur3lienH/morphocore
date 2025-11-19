import torch
import pytest
from .utils import dispatch_operation


class TestCPUvsCUDA:
    """Compare CPU and CUDA implementations to ensure consistency."""
    
    @pytest.mark.parametrize("input_size,kernel_size", [
        ((1, 1, 4, 4), (1, 1, 3, 3)),
        ((2, 3, 8, 8), (2, 3, 3, 3)),
        ((1, 1, 5, 7), (1, 1, 3, 3)),
    ])
    @pytest.mark.parametrize("channel_merge_mode", ['sum', 'mean', 'min', 'max'])
    def test_forward_cpu_vs_cuda(self, operation, input_size, kernel_size, channel_merge_mode):
        """Test that CPU and CUDA produce identical forward results."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available.")

        
        
        torch.manual_seed(42)
        input_tensor = torch.randn(*input_size, dtype=torch.float32)
        weight_tensor = torch.randn(*kernel_size, dtype=torch.float32)
        
        # CPU
        input_cpu = input_tensor.clone()
        weight_cpu = weight_tensor.clone()
        output_cpu = dispatch_operation(operation, input_cpu, weight_cpu, channel_merge_mode)
        
        # CUDA
        input_cuda = input_tensor.clone().cuda()
        weight_cuda = weight_tensor.clone().cuda()
        output_cuda = dispatch_operation(operation, input_cuda, weight_cuda, channel_merge_mode)
        
        # Check that operations returned valid results
        if output_cpu is None:
            pytest.skip(f"CPU operation returned None for {operation} with {channel_merge_mode}")
        if output_cuda is None:
            pytest.skip(f"CUDA operation returned None for {operation} with {channel_merge_mode}")
        
        # Compare
        print(f"\nOperation: {operation}, Merge: {channel_merge_mode}")
        print(f"Input shape: {input_size}, Kernel shape: {kernel_size}")
        print(f"CPU output:\n{output_cpu.squeeze()}")
        print(f"CUDA output:\n{output_cuda.cpu().squeeze()}")
        
        torch.testing.assert_close(
            output_cpu, 
            output_cuda.cpu(), 
            rtol=1e-5, 
            atol=1e-6,
            msg=f"CPU and CUDA forward outputs differ for {operation}"
        )
    
    @pytest.mark.parametrize("input_size,kernel_size", [
        ((1, 1, 4, 4), (1, 1, 3, 3)),
        ((1, 1, 32, 32), (1, 1, 3, 3)),
    ])
    @pytest.mark.parametrize("channel_merge_mode", ['sum', 'mean', 'min', 'max'])
    def test_backward_cpu_vs_cuda(self, operation, input_size, kernel_size, channel_merge_mode):
        """Test that CPU and CUDA produce identical gradients."""

        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available.")
        
        
        torch.manual_seed(42)
        
        # Create the data only once
        input_data = torch.randn(*input_size, dtype=torch.float32)
        weight_data = torch.randn(*kernel_size, dtype=torch.float32)
        
        # CPU - create leaf tensors
        input_cpu = input_data.clone().requires_grad_(True)
        weight_cpu = weight_data.clone().requires_grad_(True)
        output_cpu = dispatch_operation(operation, input_cpu, weight_cpu, channel_merge_mode)
        
        # Check that CPU operation returned a valid result
        if output_cpu is None:
            pytest.skip(f"CPU operation returned None for {operation} with {channel_merge_mode}")
        
        loss_cpu = output_cpu.sum()
        loss_cpu.backward()
        grad_input_cpu = input_cpu.grad.clone() if input_cpu.grad is not None else None
        grad_weight_cpu = weight_cpu.grad.clone() if weight_cpu.grad is not None else None
        
        # CUDA - create leaf tensors
        input_cuda = input_data.clone().cuda().requires_grad_(True)
        weight_cuda = weight_data.clone().cuda().requires_grad_(True)
        output_cuda = dispatch_operation(operation, input_cuda, weight_cuda, channel_merge_mode)
        
        # Check that CUDA operation returned a valid result
        if output_cuda is None:
            pytest.skip(f"CUDA operation returned None for {operation} with {channel_merge_mode}")
        
        loss_cuda = output_cuda.sum()
        loss_cuda.backward()
        grad_input_cuda = input_cuda.grad.clone().cpu() if input_cuda.grad is not None else None
        grad_weight_cuda = weight_cuda.grad.clone().cpu() if weight_cuda.grad is not None else None
        
        # Check that gradients exist
        if grad_input_cpu is None or grad_input_cuda is None:
            pytest.skip(f"Gradients for input are None for {operation} with {channel_merge_mode}")
        if grad_weight_cpu is None or grad_weight_cuda is None:
            pytest.skip(f"Gradients for weight are None for {operation} with {channel_merge_mode}")
        
        # Compare outputs
        print(f"\nOperation: {operation}, Merge: {channel_merge_mode}")
        print(f"CPU output:\n{output_cpu.squeeze()}")
        print(f"CUDA output:\n{output_cuda.cpu().squeeze()}")
        
        torch.testing.assert_close(
            output_cpu, 
            output_cuda.cpu(), 
            rtol=1e-4, 
            atol=1e-6,
            msg=f"CPU and CUDA outputs differ"
        )
        
        # Compare gradients
        print(f"\nCPU grad_input:\n{grad_input_cpu.squeeze()}")
        print(f"CUDA grad_input:\n{grad_input_cuda.squeeze()}")
        print(f"\nCPU grad_weight:\n{grad_weight_cpu.squeeze()}")
        print(f"CUDA grad_weight:\n{grad_weight_cuda.squeeze()}")
        
        torch.testing.assert_close(
            grad_input_cpu, 
            grad_input_cuda, 
            rtol=1e-4, 
            atol=1e-6,
            msg=f"CPU and CUDA grad_input differ"
        )
        
        torch.testing.assert_close(
            grad_weight_cpu, 
            grad_weight_cuda, 
            rtol=1e-4, 
            atol=1e-6,
            msg=f"CPU and CUDA grad_weight differ"
        )
