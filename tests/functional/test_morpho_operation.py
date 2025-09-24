import torch
import numpy as np
import morphocore.functional as F
import pytest
from .utils import correct_morphology, dispatch_operation
from .conftest import device, operation

class TestMorphologicalOperations:

    def test_morphology_4d(self, device: str, operation: str):
        image_np = np.random.randn(64, 64)
    
        kernel = np.random.randn(3, 3)
        scipy_res = correct_morphology(image_np, kernel, operation)

        image_torch = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0).to(device)
        selem_torch = torch.from_numpy(kernel.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        torch_res = dispatch_operation(operation, image_torch, selem_torch).squeeze(0).squeeze(0).cpu()

        if operation.startswith('S'):
            np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-3, atol=1e-3)
        else:  
            np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-5, atol=1e-6)

    def test_morphology_2d(self, device: str, operation: str):
        image_np = np.random.randn(64, 64)
    
        kernel = np.random.randn(3, 3)
        scipy_res = correct_morphology(image_np, kernel, operation)

        image_torch = torch.from_numpy(image_np).float().to(device)
        selem_torch = torch.from_numpy(kernel.astype(np.float32)).to(device)

        torch_res = dispatch_operation(operation, image_torch, selem_torch).cpu()

        if operation.startswith('S'):
            np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-3, atol=1e-3)
        else:  
            np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-5, atol=1e-6)

    def test_morphology_weird_size(self, device: str, operation: str):
        image_np = np.random.randn(13, 4)
    
        kernel = np.random.randn(4, 4)
        scipy_res = correct_morphology(image_np, kernel, operation)

        image_torch = torch.from_numpy(image_np).float().to(device)
        selem_torch = torch.from_numpy(kernel.astype(np.float32)).to(device)

        torch_res = dispatch_operation(operation, image_torch, selem_torch).cpu()

        if operation.startswith('S'):
            np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-3, atol=1e-3)
        else:  
            np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-5, atol=1e-6)

    def test_batch_morphology(self, device: str, operation: str):
        batch_size = 2
        
        images = []
        for i in range(batch_size):
            image = np.random.randn(64, 64)
            images.append(image)
        
        image_np = np.stack(images) 
        
        kernel = np.random.randn(3, 3)
        
        scipy_res = np.stack([correct_morphology(image_np[i], kernel, operation) for i in range(batch_size)])
        
        image_torch = torch.from_numpy(image_np).float().unsqueeze(1).to(device)
        selem_torch = torch.from_numpy(kernel.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        
        torch_res = dispatch_operation(operation, image_torch, selem_torch).squeeze(1).cpu()

        if operation.startswith('S'):
            np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-3, atol=1e-3)
        else:  
            np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-5, atol=1e-6)

    def test_one_channel_to_many(self, device: str, operation: str):
        """Test applying multiple different kernels to a single-channel image to produce multiple outputs."""
        batch_size = 1
        num_output_channels = 3
        height, width = 32, 32
        kernel_size = 3
        
        image_channel = np.random.randn(height, width)
        
        kernels = []
        for o in range(num_output_channels):
            kernel = np.random.randn(kernel_size, kernel_size)
            kernels.append(kernel)
        
        expected_channels = []
        for o in range(num_output_channels):
            channel_result = correct_morphology(image_channel, kernels[o], operation)
            expected_channels.append(channel_result)
        scipy_res = np.stack(expected_channels)

        image_torch = torch.from_numpy(image_channel).float().unsqueeze(0).unsqueeze(0).to(device)
        
        weight_tensor = torch.zeros(num_output_channels, 1, kernel_size, kernel_size).to(device)
        for o in range(num_output_channels):
            weight_tensor[o, 0, :, :] = torch.from_numpy(kernels[o].astype(np.float32)).to(device)
        

        torch_res = dispatch_operation(operation, image_torch, weight_tensor).squeeze(0).cpu()
        
        print(f"Device: {device}, Operation: {operation}")
        print(f"Input shape: {image_torch.shape}")        # (1, 1, 32, 32)
        print(f"Weight shape: {weight_tensor.shape}")     # (3, 1, 3, 3)
        print(f"Output shape: {torch_res.shape}")         # (3, 32, 32)
        print(f"Expected shape: {scipy_res.shape}")       
        
        if operation.startswith('S'):
            np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-3, atol=1e-3)
        else:  
            np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-5, atol=1e-6)

    def test_many_channel_to_one(self, device: str, operation: str):
        """Test applying different structuring elements to different channels, producing single output via accumulation."""
        batch_size = 1
        num_channels = 3
        height, width = 32, 32
        kernel_size = 3
        
        image_channels = []
        for c in range(num_channels):
            channel = np.random.randn(height, width)
            image_channels.append(channel)
        
        kernels = []
        for c in range(num_channels):
            kernel = np.random.randn(kernel_size, kernel_size)
            kernels.append(kernel)
        
        channel_results = []
        for c in range(num_channels):
            channel_result = correct_morphology(image_channels[c], kernels[c], operation)
            channel_results.append(channel_result)
        
        if operation == 'dilation' or operation == 'Sdilation':
            scipy_res = np.sum(channel_results, axis=0)
        else:
            scipy_res = np.sum(channel_results, axis=0)

        image_torch = torch.from_numpy(np.stack(image_channels)).float().unsqueeze(0).to(device)
        
        weight_tensor = torch.zeros(1, num_channels, kernel_size, kernel_size).to(device)
        for c in range(num_channels):
            weight_tensor[0, c, :, :] = torch.from_numpy(kernels[c].astype(np.float32)).to(device)
        
        print(f"Device: {device}, Operation: {operation}")
        print(f"Input shape: {image_torch.shape}")
        print(f"Weight shape: {weight_tensor.shape}")

        torch_res = dispatch_operation(operation, image_torch, weight_tensor).squeeze().cpu()

        print(f"Output shape: {torch_res.shape}")
        print(f"Expected shape: {scipy_res.shape}")
        
        if operation.startswith('S'):
            np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-3, atol=1e-3)
        else:  
            np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-5, atol=1e-6)

    def test_many_channel_to_many(self, device, operation):
        """Test applying different structuring elements, with each output getting contributions from ALL inputs."""
        batch_size = 1
        num_channels = 3
        height, width = 32, 32
        kernel_size = 3
        
        image_channels = []
        for c in range(num_channels):
            channel = np.random.randn(height, width)
            image_channels.append(channel)
        
        kernels = np.random.randn(num_channels, num_channels, kernel_size, kernel_size).astype(np.float32)
        
        expected_channels = []
        for o in range(num_channels):
            output_result = np.zeros((height, width))
            
            for i in range(num_channels):
                channel_contribution = correct_morphology(image_channels[i], kernels[o, i], operation)
                output_result += channel_contribution
                
            expected_channels.append(output_result)
        
        scipy_res = np.stack(expected_channels)
        
        image_torch = torch.from_numpy(np.stack(image_channels)).float().unsqueeze(0).to(device)  # (1, 3, H, W)
        weight_tensor = torch.from_numpy(kernels).to(device)
        
        print(f"Device: {device}, Operation: {operation}")
        print(f"Input shape: {image_torch.shape}") 
        print(f"Weight shape: {weight_tensor.shape}")

        torch_res = dispatch_operation(operation, image_torch, weight_tensor).squeeze(0).cpu()

        print(f"Output shape: {torch_res.shape}")
        print(f"Expected shape: {scipy_res.shape}")

        if operation.startswith('S'):
            np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-3, atol=1e-3)
        else:  
            np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-5, atol=1e-6)



class TestMorphologyBackward:
    
    def test_grad(self, device, operation):
        input_test = torch.tensor([[[[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0], 
                           [7.0, 8.0, 9.0]]]], dtype=torch.float32, requires_grad=True).to(device)

        weight_test = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.4, 0.5, 0.6],
                                [0.7, 0.8, 0.9]]]], dtype=torch.float32).to(device)
        
        print(f"Device: {device}, Operation: {operation}")
        print("Input:")
        print(input_test.squeeze().cpu())
        print("Weight:")
        print(weight_test.squeeze().cpu())
        
        if operation == 'dilation':
            func = lambda x, w: F.dilation(x, w)
        else:
            func = lambda x, w: F.erosion(x, w)
        
        result = torch.autograd.gradcheck(
            func,
            (input_test, weight_test),
            eps=1e-4,           
            atol=1e-3,          
            rtol=1e-2,         
            nondet_tol=1e-5,    
            fast_mode=True    
        )
        
        assert result, f"Gradient check failed for morphological {operation}"



