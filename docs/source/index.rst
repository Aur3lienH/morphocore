.. Morphocore documentation master file

Morphocore Documentation
========================

**Morphocore** is a high-performance Python library that brings mathematical morphology to deep learning. 
It provides GPU-accelerated, differentiable morphological operations that can be integrated 
into PyTorch neural networks.

Key Features
------------

* **Differentiable Operations**: All operations support backpropagation for end-to-end training
* **GPU Acceleration**: CUDA-optimized implementations for high performance
* **Flexible Architecture**: Both functional and module-based (nn) interfaces
* **Learnable Morphology**: Trainable structuring elements and interpolation parameters

Morphological Operations
------------------------

Morphocore implements the fundamental morphological operations:

* **Dilation** (⊕): Expands bright regions
* **Erosion** (⊖): Shrinks bright regions  
* **EMorph**: Exact interpolation between erosion and dilation
* **SMorph**: Smooth approximation for better gradient flow

Quick Start
-----------

Installation
^^^^^^^^^^^^

.. code-block:: bash

   pip install morphocore --no-build-isolation

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   import torch
   from morphocore.nn import Dilation

   # Create a learnable dilation layer
   layer = Dilation(in_channels=3, out_channels=16, kernel_size=5)
   
   # Forward pass
   x = torch.randn(1, 3, 32, 32)
   output = layer(x)  # Shape: (1, 16, 32, 32)
   
   # Training
   loss = criterion(output, target)
   loss.backward()
   optimizer.step()

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
