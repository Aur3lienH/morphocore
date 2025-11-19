import torch
import torch.nn as nn
import torch.optim as optim
import morphocore.nn as Mnn
from morphocore.functional import dilation, erosion
import time

# Check for CUDA availability
device = "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def test_smorph_convergence_to_dilation():
    """Test that Smorph can learn to behave like a dilation"""
    
    start_time = time.time()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Define target structuring element (1 channel, 1 input channel, 3x3 kernel)
    target_se = torch.tensor([[[[0.5, 1.0, 0.5],
                                [1.0, 2.0, 1.0],
                                [0.5, 1.0, 0.5]]]], dtype=torch.float32).to(device)
    
    # Create Smorph layer with large positive alpha (dilation-like)
    smorph_layer = Mnn.SMorph(
        in_channels=1, 
        out_channels=1, 
        kernel_size=(3, 3),
        channel_merge_mode="sum"
    ).to(device)
    
    # Generate random input images for training
    n_samples = 1024  # Increased for better batching
    batch_size = 16
    inputs = torch.rand(n_samples, 1, 16, 16, device=device) * 10
    
    # Compute target outputs using true dilation in batches
    print("Computing target outputs...")
    with torch.no_grad():
        targets = torch.cat([
            torch.stack([dilation(img.unsqueeze(0), target_se, "sum") 
                        for img in batch]).squeeze(1)
            for batch in inputs.split(batch_size)
        ])
    
    # Training with batching
    optimizer = optim.Adam(smorph_layer.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Compile model for faster execution (PyTorch 2.0+)
    try:
        smorph_layer = torch.compile(smorph_layer)
        print("Model compiled with torch.compile()")
    except:
        print("torch.compile not available, continuing without compilation")
    
    print("Training Smorph to converge to Dilation...")
    num_epochs = 500  # Reduced epochs due to better batching
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Process in batches
        for i in range(0, n_samples, batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = smorph_layer(batch_inputs)
            loss = criterion(outputs, batch_targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
    
    training_time = time.time() - start_time
    print(f"\nTraining time: {training_time:.2f}s")
    
    # Verify convergence
    print("\nTarget structuring element:")
    print(target_se.squeeze().cpu())
    print("\nLearned structuring element:")
    print(smorph_layer.weight.data.squeeze().cpu())
    print(f"Alpha value after training: {smorph_layer.alpha.item():.6f}")
    
    # Test on new data with batching
    test_batch = torch.rand(batch_size, 1, 16, 16, device=device) * 10
    with torch.no_grad():
        smorph_output = smorph_layer(test_batch)
        target_output = torch.stack([dilation(img.unsqueeze(0), target_se, "sum") 
                                    for img in test_batch]).squeeze(1)
        final_error = torch.abs(smorph_output - target_output).mean()
    
    print(f"\nFinal test error: {final_error.item():.6f}")
    assert final_error < 0.5, "Smorph did not converge to dilation"
    print("✓ Smorph successfully converged to dilation!")


def test_smorph_convergence_to_erosion():
    """Test that Smorph can learn to behave like an erosion"""
    
    start_time = time.time()
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Define target structuring element
    target_se = torch.tensor([[[[0.3, 0.8, 0.3],
                                [0.8, 1.5, 0.8],
                                [0.3, 0.8, 0.3]]]], dtype=torch.float32).to(device)
    
    # Create Smorph layer
    smorph_layer = Mnn.SMorph(
        in_channels=1, 
        out_channels=1, 
        kernel_size=(3, 3)
    ).to(device)
    
    # Generate training data
    n_samples = 1024
    batch_size = 32
    inputs = torch.rand(n_samples, 1, 16, 16, device=device) * 10
    
    # Compute target outputs using true erosion in batches
    print("\n" + "="*60)
    print("Computing target outputs...")
    with torch.no_grad():
        targets = torch.cat([
            torch.stack([erosion(img.unsqueeze(0), target_se, "sum") 
                        for img in batch]).squeeze(1)
            for batch in inputs.split(batch_size)
        ])
    
    # Training
    optimizer = optim.Adam(smorph_layer.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Compile model
    try:
        smorph_layer = torch.compile(smorph_layer)
        print("Model compiled with torch.compile()")
    except:
        print("torch.compile not available, continuing without compilation")
    
    print("Training Smorph to converge to Erosion...")
    num_epochs = 500  # Reduced epochs
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Process in batches
        for i in range(0, n_samples, batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = smorph_layer(batch_inputs)
            loss = criterion(outputs, batch_targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
    
    training_time = time.time() - start_time
    print(f"\nTraining time: {training_time:.2f}s")
    
    # Verify convergence
    print("\nTarget structuring element:")
    print(target_se.squeeze().cpu())
    print("\nLearned structuring element:")
    print(smorph_layer.weight.data.squeeze().cpu())
    print(f"Alpha value after training: {smorph_layer.alpha.item():.6f}")
    
    # Test on new data
    test_batch = torch.rand(batch_size, 1, 16, 16, device=device) * 10
    with torch.no_grad():
        smorph_output = smorph_layer(test_batch)
        target_output = torch.stack([erosion(img.unsqueeze(0), target_se, "sum") 
                                    for img in test_batch]).squeeze(1)
        final_error = torch.abs(smorph_output - target_output).mean()
    
    print(f"\nFinal test error: {final_error.item():.6f}")
    assert final_error < 0.5, "Smorph did not converge to erosion"
    print("✓ Smorph successfully converged to erosion!")


if __name__ == "__main__":
    test_smorph_convergence_to_dilation()
    test_smorph_convergence_to_erosion()
    print("\n" + "="*60)
    print("All tests passed! ✓")