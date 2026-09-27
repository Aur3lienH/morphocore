Installation
============

Morphocore can be installed from PyPI or built from source.

From PyPI (Recommended)
-----------------------

The simplest way to install Morphocore is via pip:

.. code-block:: bash

   pip install morphocore

Requirements
^^^^^^^^^^^^

For pre-compiled packages from PyPI:

* Python >= 3.11
* PyTorch >= 2.6
* CUDA >= 12.3 (for GPU support)

.. note::
   If you encounter compatibility issues with the pre-compiled version, 
   please compile from source as described below.

From Source
-----------

Building from source gives you more flexibility and ensures compatibility with your environment.

Prerequisites
^^^^^^^^^^^^^

* Git
* Python >= 3.11
* PyTorch >= 2.0
* CUDA Toolkit (for GPU support, optional)
* C++ compiler (GCC >= 7.0 or equivalent)

Installation Steps
^^^^^^^^^^^^^^^^^^

**Standard Installation (with CUDA)**

1. Clone the repository:

.. code-block:: bash

   git clone git@gitlab.lre.epita.fr:aurelien.hurand/morphocore.git
   cd morphocore

2. Install with CUDA support:

.. code-block:: bash

   pip install . --no-build-isolation

**CPU-Only Installation (without CUDA)**

If you don't have a GPU or want to compile without CUDA support:

.. code-block:: bash

   git clone git@gitlab.lre.epita.fr:aurelien.hurand/morphocore.git
   cd morphocore
   MORPHO_WITH_CUDA=0 pip install . --no-build-isolation

This will compile only the CPU version, which:

* Doesn't require CUDA Toolkit installation
* Works on machines without NVIDIA GPUs
* Runs on PyTorch's CPU thread pool (set the thread count with ``torch.set_num_threads``)
* Uses AVX2 automatically when the processor supports it, and baseline x86-64 code otherwise;
  set ``MORPHOCORE_DISABLE_AVX2=1`` to force the baseline code
* Is faster to compile

.. note::
   The CPU version will still work but will be significantly slower for large tensors.
   GPU acceleration provides 10-100x speedup depending on the operation and tensor size.

Development Installation
^^^^^^^^^^^^^^^^^^^^^^^^

For development, install in editable mode with test dependencies:

.. code-block:: bash

   # With CUDA
   pip install -e . --no-build-isolation
   pip install pytest pytest-cov
   
   # Without CUDA
   MORPHO_WITH_CUDA=0 pip install -e . --no-build-isolation
   pip install pytest pytest-cov

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

MorphoCore supports the following environment variables during installation:

* **MORPHO_WITH_CUDA**: Set to ``0`` to disable CUDA compilation (default: ``1``)
* **TORCH_VERSION**: Specify PyTorch version for compatibility (auto-detected by default)

Examples:

.. code-block:: bash

   # Force CPU-only build
   MORPHO_WITH_CUDA=0 pip install .
   
   # Specify PyTorch version
   TORCH_VERSION=2.0.1 pip install .
   
   # Combine both
   MORPHO_WITH_CUDA=0 TORCH_VERSION=2.0.1 pip install .

Verifying Installation
----------------------

To verify that Morphocore is correctly installed:

.. code-block:: python

   import morphocore
   import torch
   from morphocore.nn import Dilation

   # Check if CUDA is available
   print(f"CUDA available: {torch.cuda.is_available()}")
   
   # Create a simple layer
   layer = Dilation(3, 16, kernel_size=5)
   x = torch.randn(1, 3, 32, 32)
   output = layer(x)
   
   print(f"Output shape: {output.shape}")
   print("Installation successful!")

Troubleshooting
---------------

CUDA Compatibility Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you encounter CUDA-related errors, ensure that:

* Your PyTorch version is compiled with the same CUDA version as your system
* The CUDA toolkit is properly installed and in your PATH

Check your PyTorch CUDA version:

.. code-block:: python

   import torch
   print(torch.version.cuda)

Compilation Errors
^^^^^^^^^^^^^^^^^^

If compilation fails:

1. Ensure you have a compatible C++ compiler installed
2. For CUDA builds, verify that CUDA toolkit is properly installed
3. Try updating setuptools: ``pip install --upgrade setuptools wheel``
4. **If CUDA is causing issues**, try CPU-only build: ``MORPHO_WITH_CUDA=0 pip install .``

Common CUDA compilation issues:

.. code-block:: bash

   # Error: nvcc not found
   # Solution: Install CUDA toolkit or use CPU-only build
   MORPHO_WITH_CUDA=0 pip install .
   
   # Error: CUDA version mismatch
   # Solution: Match PyTorch CUDA version or disable CUDA
   python -c "import torch; print(torch.version.cuda)"  # Check PyTorch CUDA version
   MORPHO_WITH_CUDA=0 pip install .  # Or use CPU build

Import Errors
^^^^^^^^^^^^^

If you get import errors after installation:

1. Ensure you're not in the source directory when importing
2. Check that the package is installed: ``pip list | grep morphocore``