API Reference
=============

MorphoCore provides two interfaces for morphological operations:

1. **Functional API** (`morphocore.functional`): Stateless functions for direct operations
2. **Module API** (`morphocore.nn`): Learnable layers for neural networks

Functional API
--------------

The functional API provides stateless operations that can be called directly with tensors.

Basic Morphological Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dilation
""""""""

.. autofunction:: morphocore.functional.dilation
   :no-index:

Erosion
"""""""

.. autofunction:: morphocore.functional.erosion
   :no-index:

Composite Operations
^^^^^^^^^^^^^^^^^^^^

Opening
"""""""

.. autofunction:: morphocore.functional.opening
   :no-index:

Closing
"""""""

.. autofunction:: morphocore.functional.closing
   :no-index:

Exact Interpolation
^^^^^^^^^^^^^^^^^^^

SigmaLayer
""""""""""

.. autofunction:: morphocore.functional.sigma_layer
   :no-index:

Smooth Approximation
^^^^^^^^^^^^^^^^^^^^

SMorph
""""""

.. autofunction:: morphocore.functional.smorph
   :no-index:



Module API (nn)
---------------

The module API provides learnable layers that inherit from `torch.nn.Module`.

Basic Morphological Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dilation Layer
""""""""""""""

.. autoclass:: morphocore.nn.Dilation
   :members:
   :special-members: __init__
   :show-inheritance:
   :no-index:

Erosion Layer
"""""""""""""

.. autoclass:: morphocore.nn.Erosion
   :members:
   :special-members: __init__
   :show-inheritance:
   :no-index:

Composite Layers
^^^^^^^^^^^^^^^^

Opening Layer
"""""""""""""

.. autoclass:: morphocore.nn.Opening
   :members:
   :special-members: __init__
   :show-inheritance:
   :no-index:

Closing Layer
"""""""""""""

.. autoclass:: morphocore.nn.Closing
   :members:
   :special-members: __init__
   :show-inheritance:
   :no-index:

Exact Interpolation Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^

SigmaLayer
""""""""""

.. autoclass:: morphocore.nn.SigmaLayer
   :members:
   :special-members: __init__
   :show-inheritance:
   :no-index:

Smooth Approximation Layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SMorph Layer
""""""""""""

.. autoclass:: morphocore.nn.SMorph
   :members:
   :special-members: __init__
   :show-inheritance:
   :no-index: