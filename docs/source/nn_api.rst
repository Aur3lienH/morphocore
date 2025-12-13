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

EMorph Layer
""""""""""""

.. autoclass:: morphocore.nn.EMorph
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
