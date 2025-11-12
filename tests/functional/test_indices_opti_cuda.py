import torch
import pytest
import morphocore.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_forward_save_indices_consistency():
    """Test that forward pass produces identical results with and without save_indices"""
    device = 'cuda'
    torch.manual_seed(42)
    
    # Test both operations and merge modes
    test_cases = [
        ('dilation', 'max'),
        ('dilation', 'min'),
        ('erosion', 'max'),
        ('erosion', 'min'),
    ]
    
    for operation, merge_mode in test_cases:
        print(f"\n=== Testing {operation} with {merge_mode} ===")
        
        # Create test data
        img = torch.randn(2, 4, 16, 16, device=device, dtype=torch.float32)
        weight = torch.randn(4, 4, 5, 5, device=device, dtype=torch.float32)
        
        # Forward WITHOUT indices
        if operation == 'dilation':
            out_no_idx = F.dilation(img, weight, channel_merge_mode=merge_mode, save_indices=False)
        else:
            out_no_idx = F.erosion(img, weight, channel_merge_mode=merge_mode, save_indices=False)
        
        # Forward WITH indices
        if operation == 'dilation':
            out_with_idx = F.dilation(img, weight, channel_merge_mode=merge_mode, save_indices=True)
        else:
            out_with_idx = F.erosion(img, weight, channel_merge_mode=merge_mode, save_indices=True)
        
        # Compare
        max_diff = (out_no_idx - out_with_idx).abs().max().item()
        rel_error = max_diff / (out_no_idx.abs().max().item() + 1e-8)
        
        print(f"Max absolute difference: {max_diff:.2e}")
        print(f"Relative error: {rel_error:.2e}")
        print(f"Outputs match: {torch.allclose(out_no_idx, out_with_idx, rtol=1e-5, atol=1e-6)}")
        
        assert torch.allclose(out_no_idx, out_with_idx, rtol=1e-5, atol=1e-6), \
            f"Forward outputs differ for {operation} with {merge_mode}: max_diff={max_diff:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_backward_save_indices_consistency():
    """Test that backward pass produces identical gradients with and without save_indices"""
    device = 'cuda'
    torch.manual_seed(42)
    
    # Test both operations and merge modes
    test_cases = [
        ('dilation', 'max'),
        ('dilation', 'min'),
        ('erosion', 'max'),
        ('erosion', 'min'),
    ]
    
    for operation, merge_mode in test_cases:
        print(f"\n=== Testing {operation} with {merge_mode} ===")
        
        # Create test data
        img_data = torch.randn(2, 4, 16, 16, device=device, dtype=torch.float32)
        weight_data = torch.randn(4, 4, 5, 5, device=device, dtype=torch.float32)
        
        # Test WITHOUT indices
        img_no_idx = img_data.clone().requires_grad_(True)
        weight_no_idx = weight_data.clone().requires_grad_(True)
        
        if operation == 'dilation':
            out_no_idx = F.dilation(img_no_idx, weight_no_idx, channel_merge_mode=merge_mode, save_indices=False)
        else:
            out_no_idx = F.erosion(img_no_idx, weight_no_idx, channel_merge_mode=merge_mode, save_indices=False)
        
        out_no_idx.sum().backward()
        grad_input_no_idx = img_no_idx.grad.clone()
        grad_weight_no_idx = weight_no_idx.grad.clone()
        
        # Test WITH indices
        img_with_idx = img_data.clone().requires_grad_(True)
        weight_with_idx = weight_data.clone().requires_grad_(True)
        
        if operation == 'dilation':
            out_with_idx = F.dilation(img_with_idx, weight_with_idx, channel_merge_mode=merge_mode, save_indices=True)
        else:
            out_with_idx = F.erosion(img_with_idx, weight_with_idx, channel_merge_mode=merge_mode, save_indices=True)
        
        out_with_idx.sum().backward()
        grad_input_with_idx = img_with_idx.grad.clone()
        grad_weight_with_idx = weight_with_idx.grad.clone()
        
        # Compare input gradients
        max_diff_input = (grad_input_no_idx - grad_input_with_idx).abs().max().item()
        rel_error_input = max_diff_input / (grad_input_no_idx.abs().max().item() + 1e-8)
        
        # Compare weight gradients
        max_diff_weight = (grad_weight_no_idx - grad_weight_with_idx).abs().max().item()
        rel_error_weight = max_diff_weight / (grad_weight_no_idx.abs().max().item() + 1e-8)
        
        print(f"Input gradient - Max diff: {max_diff_input:.2e}, Rel error: {rel_error_input:.2e}")
        print(f"Weight gradient - Max diff: {max_diff_weight:.2e}, Rel error: {rel_error_weight:.2e}")
        
        # Sparsity info
        input_nonzero_no_idx = (grad_input_no_idx != 0).sum().item()
        input_nonzero_with_idx = (grad_input_with_idx != 0).sum().item()
        print(f"Input gradient non-zeros: WITHOUT={input_nonzero_no_idx}, WITH={input_nonzero_with_idx}")
        
        weight_nonzero_no_idx = (grad_weight_no_idx != 0).sum().item()
        weight_nonzero_with_idx = (grad_weight_with_idx != 0).sum().item()
        print(f"Weight gradient non-zeros: WITHOUT={weight_nonzero_no_idx}, WITH={weight_nonzero_with_idx}")
        
        assert torch.allclose(grad_input_no_idx, grad_input_with_idx, rtol=1e-4, atol=1e-5), \
            f"Input gradients differ for {operation} with {merge_mode}: max_diff={max_diff_input:.2e}"
        
        assert torch.allclose(grad_weight_no_idx, grad_weight_with_idx, rtol=1e-4, atol=1e-5), \
            f"Weight gradients differ for {operation} with {merge_mode}: max_diff={max_diff_weight:.2e}"