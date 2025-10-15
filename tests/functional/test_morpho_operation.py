import torch
import os
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
        
        # print(f"Device: {device}, Operation: {operation}")
        # print(f"Input shape: {image_torch.shape}")        # (1, 1, 32, 32)
        # print(f"Weight shape: {weight_tensor.shape}")     # (3, 1, 3, 3)
        # print(f"Output shape: {torch_res.shape}")         # (3, 32, 32)
        # print(f"Expected shape: {scipy_res.shape}")       
        
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
        
        # print(f"Device: {device}, Operation: {operation}")
        # print(f"Input shape: {image_torch.shape}")
        # print(f"Weight shape: {weight_tensor.shape}")

        torch_res = dispatch_operation(operation, image_torch, weight_tensor).squeeze().cpu()

        # print(f"Output shape: {torch_res.shape}")
        # print(f"Expected shape: {scipy_res.shape}")
        
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
        
        # print(f"Device: {device}, Operation: {operation}")
        # print(f"Input shape: {image_torch.shape}") 
        # print(f"Weight shape: {weight_tensor.shape}")

        torch_res = dispatch_operation(operation, image_torch, weight_tensor).squeeze(0).cpu()

        # print(f"Output shape: {torch_res.shape}")
        # print(f"Expected shape: {scipy_res.shape}")

        if operation.startswith('S'):
            np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-3, atol=1e-3)
        else:  
            np.testing.assert_allclose(torch_res, scipy_res, rtol=1e-5, atol=1e-6)



class TestMorphologyBackward:
    
    def test_grad_deterministic(self, device, operation):
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
        
        func = lambda x, w: dispatch_operation(operation, x, w)
        
        if operation.startswith('S'):
            tolerance = 1e-2
        else:
            tolerance = 1e-3
        result = torch.autograd.gradcheck(
            func,
            (input_test, weight_test),
            eps=1e-4,           
            atol=tolerance,          
            rtol=tolerance * 10,         
            nondet_tol=1e-5,    
            fast_mode=False    
        )
        
        assert result, f"Gradient check failed for morphological {operation}"

    @pytest.mark.parametrize("input_size,kernel_size", [
        ((1, 1, 5, 5), (1, 1, 3, 3)),
        ((1, 1, 7, 7), (1, 1, 5, 5)),
        
        ((1, 1, 6, 8), (1, 1, 3, 5)),    # kernel 3x5
        ((1, 1, 8, 6), (1, 1, 5, 3)),    # kernel 5x3
        ((1, 1, 7, 9), (1, 1, 3, 7)),    # kernel 3x7
        
        ((1, 1, 4, 8), (1, 1, 3, 3)),    # image 4x8
        ((1, 1, 8, 4), (1, 1, 3, 3)),    # image 8x4
        ((1, 1, 6, 10), (1, 1, 3, 3)),   # image 6x10
        
        ((2, 1, 4, 4), (1, 1, 3, 3)),
        ((3, 1, 5, 5), (1, 1, 3, 3)),
        ((2, 1, 6, 8), (1, 1, 3, 5)),    # batch + rectangular
        
        ((1, 2, 6, 6), (1, 2, 3, 3)),
        ((1, 3, 5, 7), (1, 3, 3, 3)),
        ((1, 4, 6, 6), (1, 4, 5, 5)),
        
        ((1, 1, 6, 6), (3, 1, 3, 3)),
        ((1, 1, 8, 8), (4, 1, 3, 3)),
        ((1, 1, 5, 9), (2, 1, 3, 5)),    # rectangular + multi-output
        
        ((1, 2, 6, 6), (3, 2, 3, 3)),
        ((1, 3, 7, 7), (2, 3, 5, 5)),
        ((1, 2, 8, 8), (4, 2, 5, 5)),    # rectangular + multi-channel
        
        ((2, 2, 5, 5), (1, 2, 3, 3)),
        ((3, 1, 4, 6), (2, 1, 3, 3)),
        ((2, 3, 6, 8), (2, 3, 3, 5)),
        
        # ((1, 1, 9, 9), (1, 1, 7, 7)), // Issue here for the backward on double because of the size of he shared memory TODO: fix
        ((1, 1, 10, 8), (1, 1, 5, 7)),
        ((1, 2, 8, 8), (1, 2, 6, 6)),
        
        ((1, 1, 3, 3), (1, 1, 3, 3)),    # kernel == image
        ((1, 1, 4, 3), (1, 1, 3, 3)),    # image à peine plus grande
        ((1, 1, 3, 5), (1, 1, 3, 3)),    # image rectangulaire petite
        
        ((1, 1, 12, 4), (1, 1, 3, 3)),   # image très rectangulaire
        ((1, 1, 4, 12), (1, 1, 3, 3)),   # image très rectangulaire (inverse)
        ((1, 1, 8, 8), (1, 1, 7, 3)),    # kernel très rectangulaire
        ((1, 1, 8, 8), (1, 1, 3, 7)),    # kernel très rectangulaire (inverse)
    ])
    @pytest.mark.parametrize("channel_merge_mode", ['sum', 'mean', 'min', 'max'])
    def test_grad_random(self, device, operation, input_size, kernel_size, channel_merge_mode):
        if operation.startswith('S') and channel_merge_mode in ['min', 'max']:
            pytest.skip(f"channel_merge_mode '{channel_merge_mode}' not applicable for {operation}")
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        os.environ['OMP_NUM_THREADS'] = '1'
        torch.manual_seed(42)
        input_test = torch.randn(*input_size, dtype=torch.float64, requires_grad=True).to(device)
        input_test = input_test * 10 + torch.arange(input_test.numel(), dtype=torch.float64, device=device).reshape(input_test.shape)

        weight_test = torch.randn(*kernel_size, dtype=torch.float64).to(device)
        weight_test = weight_test * 10 + torch.arange(weight_test.numel(), dtype=torch.float64, device=device).reshape(weight_test.shape)
        
        # print(f"Device: {device}, Operation: {operation}")
        # print(f"Input shape: {input_test.shape}")
        # print(f"Weight shape: {weight_test.shape}")
        
        func = lambda x, w: dispatch_operation(operation, x, w, channel_merge_mode)
        
        #if operation.startswith('S'):
        tolerance = 1e-2
        
        result = torch.autograd.gradcheck(
            func,
            (input_test, weight_test),
            eps=1e-4,           
            atol=tolerance,          
            rtol=tolerance * 10,         
            nondet_tol=1e-5,    
            fast_mode=False    
        )
        
        assert result, f"Gradient check failed for morphological {operation} with random input"

    @pytest.mark.parametrize("alpha", [-1000, -10.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 10.0, 1000.0])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_grad_smorph(self, device, alpha):
        operation = 'Smorph'

        
        input_test = torch.randn(1, 3, 8, 6, dtype=torch.float64, requires_grad=True).to(device)
        weight_test = torch.randn(1, 3, 4, 3, dtype=torch.float64, requires_grad=True).to(device)
        
        alpha_test = torch.tensor(alpha, dtype=torch.float64, requires_grad=True).to(device)
    
        
        func = lambda x, w, a: F.smorph(x, w, a)
        
        if abs(alpha) < 1e-3:
            tolerance = 1e-2
        else:
            tolerance = 1e-3
        
        result = torch.autograd.gradcheck(
            func,
            (input_test, weight_test, alpha_test),
            eps=1e-4,           
            atol=tolerance,          
            rtol=tolerance * 10,         
            nondet_tol=1e-5,    
            fast_mode=False    
        )
        
        assert result, f"Gradient check failed for morphological {operation} with alpha={alpha}"



