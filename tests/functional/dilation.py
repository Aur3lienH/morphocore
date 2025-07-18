import torch
import numpy as np
import morphocore.functional as F
import pytest

def correct_dilation(image, selem):
    pad_h, pad_w = selem.shape[0]//2, selem.shape[1]//2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=-np.inf)

    result = np.full_like(image, -np.inf)
    
    for dy in range(selem.shape[0]):
        for dx in range(selem.shape[1]):
            selem_value = selem[dy, dx]
            
            offset_y = dy - pad_h
            offset_x = dx - pad_w
            
            y_start, y_end = pad_h - offset_y, pad_h - offset_y + image.shape[0]
            x_start, x_end = pad_w - offset_x, pad_w - offset_x + image.shape[1]
            
            shifted_region = padded[y_start:y_end, x_start:x_end]
            
            candidate = shifted_region + selem_value
            
            result = np.maximum(result, candidate)
    
    return result


# Device parametrization fixture
@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return request.param


class TestDilation:

    def test_dilation_4d(self, device):
        image_np = np.random.randn(64, 64)
    
        kernel = np.random.randn(3, 3)
        scipy_res = correct_dilation(image_np, kernel)

        image_torch = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0).to(device)
        selem_torch = torch.from_numpy(kernel.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        torch_res = F.dilation(image_torch, selem_torch).squeeze(0).squeeze(0).cpu()

        np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-6, atol=1e-7)

    def test_dilation_2d(self, device):
        image_np = np.random.randn(64, 64)
    
        kernel = np.random.randn(3, 3)
        scipy_res = correct_dilation(image_np, kernel)

        image_torch = torch.from_numpy(image_np).float().to(device)
        selem_torch = torch.from_numpy(kernel.astype(np.float32)).to(device)

        torch_res = F.dilation(image_torch, selem_torch).cpu()

        np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-6, atol=1e-7)

    def test_dilation_weird_size(self, device):
        image_np = np.random.randn(13, 4)
    
        kernel = np.random.randn(4, 4)
        scipy_res = correct_dilation(image_np, kernel)

        image_torch = torch.from_numpy(image_np).float().to(device)
        selem_torch = torch.from_numpy(kernel.astype(np.float32)).to(device)

        torch_res = F.dilation(image_torch, selem_torch).cpu()

        np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-6, atol=1e-7)

    def test_batch_dilation(self, device):
        batch_size = 2
        
        images = []
        for i in range(batch_size):
            image = np.random.randn(64, 64)
            images.append(image)
        
        image_np = np.stack(images) 
        
        kernel = np.random.randn(3, 3)
        
        scipy_res = np.stack([correct_dilation(image_np[i], kernel) for i in range(batch_size)])
        
        image_torch = torch.from_numpy(image_np).float().unsqueeze(1).to(device)
        selem_torch = torch.from_numpy(kernel.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        
        torch_res = F.dilation(image_torch, selem_torch).squeeze(1).cpu()
        
        np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-6, atol=1e-7)

    def test_one_channel_to_many(self, device):
        """Test applying multiple different kernels to a single-channel image to produce multiple outputs."""
        batch_size = 1
        num_output_channels = 3
        height, width = 32, 32
        kernel_size = 3
        
        # Create single-channel image
        image_channel = np.random.randn(height, width)
        
        # Create 3 different structuring elements (one for each output channel)
        kernels = []
        for o in range(num_output_channels):
            kernel = np.random.randn(kernel_size, kernel_size)
            kernels.append(kernel)
        
        # Expected result: apply each kernel to the same input channel
        expected_channels = []
        for o in range(num_output_channels):
            channel_result = correct_dilation(image_channel, kernels[o])
            expected_channels.append(channel_result)
        scipy_res = np.stack(expected_channels)  # Shape: (3, 32, 32)
        
        # Torch tensors
        # Input: (1, 1, 32, 32)
        image_torch = torch.from_numpy(image_channel).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Weight: (3, 1, 3, 3)
        weight_tensor = torch.zeros(num_output_channels, 1, kernel_size, kernel_size).to(device)
        for o in range(num_output_channels):
            weight_tensor[o, 0, :, :] = torch.from_numpy(kernels[o].astype(np.float32)).to(device)
        
        # Apply dilation
        torch_res = F.dilation(image_torch, weight_tensor).squeeze(0).cpu()
        
        print(f"Device: {device}")
        print(f"Input shape: {image_torch.shape}")        # (1, 1, 32, 32)
        print(f"Weight shape: {weight_tensor.shape}")     # (3, 1, 3, 3)
        print(f"Output shape: {torch_res.shape}")         # (3, 32, 32)
        print(f"Expected shape: {scipy_res.shape}")       
        
        np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-6, atol=1e-7)

    def test_many_channel_to_one(self, device):
        """Test applying different structuring elements to different channels, producing single output via accumulation."""
        batch_size = 1
        num_channels = 3
        height, width = 32, 32
        kernel_size = 3
        
        # Create multi-channel image
        image_channels = []
        for c in range(num_channels):
            channel = np.random.randn(height, width)
            image_channels.append(channel)
        
        # Different structuring element for each channel
        kernels = []
        for c in range(num_channels):
            kernel = np.random.randn(kernel_size, kernel_size)
            kernels.append(kernel)
        
        # ✅ CORRECTED: Expected result with ACCUMULATION (sum across channels)
        channel_results = []
        for c in range(num_channels):
            channel_result = correct_dilation(image_channels[c], kernels[c])
            channel_results.append(channel_result)
        
        # ✅ Sum across all channel results (matching C++ accumulation behavior)
        scipy_res = np.sum(channel_results, axis=0)  # Shape: (H, W)
        
        # Torch tensors
        image_torch = torch.from_numpy(np.stack(image_channels)).float().unsqueeze(0).to(device)  # (1, 3, H, W)
        
        # Weight shape: (1, 3, K, K) - 1 output channel, 3 input channels
        weight_tensor = torch.zeros(1, num_channels, kernel_size, kernel_size).to(device)
        for c in range(num_channels):
            weight_tensor[0, c, :, :] = torch.from_numpy(kernels[c].astype(np.float32)).to(device)
        
        print(f"Device: {device}")
        print(f"Input shape: {image_torch.shape}")      # (1, 3, 32, 32)
        print(f"Weight shape: {weight_tensor.shape}")   # (1, 3, 3, 3)
        
        torch_res = F.dilation(image_torch, weight_tensor).squeeze().cpu()
        
        print(f"Output shape: {torch_res.shape}")       # (32, 32)
        print(f"Expected shape: {scipy_res.shape}")     # (32, 32)
        
        np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-6, atol=1e-7)

    def test_many_channel_to_many(self, device):
        """Test applying different structuring elements, with each output getting contributions from ALL inputs."""
        batch_size = 1
        num_channels = 3
        height, width = 32, 32
        kernel_size = 3
        
        # Create multi-channel image
        image_channels = []
        for c in range(num_channels):
            channel = np.random.randn(height, width)
            image_channels.append(channel)
        
        # Create weight tensor: (3, 3, K, K) - 3 outputs, 3 inputs each
        # Each output channel has different kernels for each input channel
        kernels = np.random.randn(num_channels, num_channels, kernel_size, kernel_size).astype(np.float32)
        
        # ✅ CORRECTED: Expected result with contributions from ALL input channels
        expected_channels = []
        for o in range(num_channels):  # For each output channel
            output_result = np.zeros((height, width))
            
            for i in range(num_channels):  # Accumulate from ALL input channels
                # Apply weight[o][i] to input[i]
                channel_contribution = correct_dilation(image_channels[i], kernels[o, i])
                output_result += channel_contribution
                
            expected_channels.append(output_result)
        
        scipy_res = np.stack(expected_channels)  # Shape: (3, H, W)
        
        # PyTorch tensors
        image_torch = torch.from_numpy(np.stack(image_channels)).float().unsqueeze(0).to(device)  # (1, 3, H, W)
        weight_tensor = torch.from_numpy(kernels).to(device)  # (3, 3, K, K)
        
        print(f"Device: {device}")
        print(f"Input shape: {image_torch.shape}")        # (1, 3, 32, 32)
        print(f"Weight shape: {weight_tensor.shape}")     # (3, 3, 3, 3)
        print(f"Weight structure: FULL (each output from ALL inputs)")
        
        torch_res = F.dilation(image_torch, weight_tensor).squeeze(0).cpu()  # (3, H, W)
        
        print(f"Output shape: {torch_res.shape}")         # (3, 32, 32)
        print(f"Expected shape: {scipy_res.shape}")       # (3, 32, 32)
        
        np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-6, atol=1e-7)

    @pytest.mark.benchmark
    @pytest.mark.parametrize("image_size", [32, 64, 128])
    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_benchmark(self, benchmark, device, image_size, kernel_size, batch_size):
        images = [np.random.randn(image_size, image_size) for _ in range(batch_size)]
        image_np = np.stack(images)
        
        kernel = np.random.randn(kernel_size, kernel_size)
        
        image_torch = torch.from_numpy(image_np).float().unsqueeze(1).to(device)
        selem_torch = torch.from_numpy(kernel.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        
        result = benchmark(F.dilation, image_torch, selem_torch)
        
        assert result.shape == (batch_size, 1, image_size, image_size)
        

class TestDilationBackward:
    
    def test_grad(self, device):
        input_test = torch.tensor([[[[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0], 
                           [7.0, 8.0, 9.0]]]], dtype=torch.float32, requires_grad=True).to(device)

        weight_test = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.4, 0.5, 0.6],
                                [0.7, 0.8, 0.9]]]], dtype=torch.float32).to(device)
        
        print(f"Device: {device}")
        print("Input:")
        print(input_test.squeeze().cpu())
        print("Weight:")
        print(weight_test.squeeze().cpu())
        
        result = torch.autograd.gradcheck(
            lambda x, w: F.dilation(x, w),
            (input_test, weight_test),
            eps=1e-4,           
            atol=1e-3,          
            rtol=1e-2,         
            nondet_tol=1e-5,    
            fast_mode=True    
        )
        
        assert result, "Gradient check failed for morphological dilation"