Quick Start Guide
=================

This guide will help you get started with MorphoCore's basic operations.

Basic Morphological Operations
-------------------------------

Dilation
^^^^^^^^

Dilation expands bright regions in the image:

.. code-block:: python

   import torch
   from morphocore.nn import Dilation

   # Create a dilation layer
   dilation = Dilation(in_channels=3, out_channels=16, kernel_size=5)
   
   x = torch.randn(1, 3, 32, 32)
   output = dilation(x)
   print(output.shape)  # torch.Size([1, 16, 32, 32])

Erosion
^^^^^^^

Erosion shrinks bright regions:

.. code-block:: python

   from morphocore.nn import Erosion

   # Create an erosion layer
   erosion = Erosion(in_channels=3, out_channels=16, kernel_size=5)
   
   x = torch.randn(1, 3, 32, 32)
   output = erosion(x)

Composite Operations
--------------------

Opening
^^^^^^^

Opening removes small objects and noise (erosion followed by dilation):

.. code-block:: python

   from morphocore.nn import Opening

   opening = Opening(in_channels=3, out_channels=16, kernel_size=5)
   output = opening(x)

Closing
^^^^^^^

Closing fills small holes (dilation followed by erosion):

.. code-block:: python

   from morphocore.nn import Closing

   closing = Closing(in_channels=3, out_channels=16, kernel_size=5)
   output = closing(x)

Advanced Operations
-------------------

SigmaLayer (Exact Interpolation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SigmaLayer provides exact interpolation between erosion and dilation:

.. code-block:: python

   from morphocore.nn import SigmaLayer

   # Create a SigmaLayer with learnable alpha
   sigma_layer = SigmaLayer(in_channels=3, out_channels=16, kernel_size=5, init_alpha=0.0)
   
   x = torch.randn(1, 3, 32, 32)
   output = sigma_layer(x)
   
   # Alpha can be learned during training
   # When alpha → +∞: operation → dilation
   # When alpha → -∞: operation → erosion

SMorph (Smooth Approximation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SMorph uses a smooth approximation for better gradient flow:

.. code-block:: python

   from morphocore.nn import SMorph

   # Create SMorph layer
   smorph = SMorph(in_channels=3, out_channels=16, kernel_size=5, init_alpha=0.0)
   
   x = torch.randn(1, 3, 32, 32)
   output = smorph(x)

Training a Morphological Network
---------------------------------

Here's a complete example of training a network with morphological layers:

.. code-block:: python

   import torch
   import torch.nn as nn
   from morphocore.nn import Dilation, Erosion, SigmaLayer

   class MorphoNet(nn.Module):
       def __init__(self):
           super().__init__()
           self.dilation = Dilation(3, 16, kernel_size=5)
           self.erosion = Erosion(16, 16, kernel_size=5)
           self.sigma_layer = SigmaLayer(16, 8, kernel_size=3, init_alpha=0.0)
           self.conv = nn.Conv2d(8, 10, kernel_size=1)
       
       def forward(self, x):
           x = self.dilation(x)
           x = self.erosion(x)
           x = self.sigma_layer(x)
           x = self.conv(x)
           return x

   # Training loop
   model = MorphoNet()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   criterion = nn.CrossEntropyLoss()

   for epoch in range(10):
       optimizer.zero_grad()
       
       # Forward pass
       x = torch.randn(8, 3, 32, 32)
       target = torch.randint(0, 10, (8,))
       
       output = model(x)
       output = output.mean(dim=(-2, -1))  # Global average pooling
       
       # Backward pass
       loss = criterion(output, target)
       loss.backward()
       optimizer.step()
       
       print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

Functional API
--------------

For more control, use the functional API:

.. code-block:: python

   from morphocore.functional import dilation, erosion, sigma_layer

   # Define structuring elements
   w = torch.ones(16, 3, 5, 5)
   alpha = torch.zeros(16, 3)
   
   # Apply operations
   x = torch.randn(1, 3, 32, 32)
   dilated = dilation(x, w)
   eroded = erosion(x, w)
   interpolated = sigma_layer(x, w, alpha)

Channel Merge Modes
-------------------

Different merge modes for multi-channel processing:

.. code-block:: python

   from morphocore.nn import Dilation

   # Sum mode (default): sum contributions from all input channels
   dil_sum = Dilation(3, 16, 5, channel_merge_mode='sum')
   
   # Max mode: take maximum across input channels
   dil_max = Dilation(3, 16, 5, channel_merge_mode='max')
   
   # Identity mode: one-to-one channel correspondence
   dil_identity = Dilation(3, 3, 5, channel_merge_mode='identity')

Padding Modes
-------------

Different padding strategies for border handling:

.. code-block:: python

   # Geodesic padding: morphologically-aware padding
   layer_geo = Dilation(3, 16, 5, padding_mode='geodesic')
   
   # Replicate padding (default): replicate border values
   layer_rep = Dilation(3, 16, 5, padding_mode='replicate')