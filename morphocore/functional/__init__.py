from .dilation import dilation
from .erosion import erosion
from .smorph import smorph
from .emorph import emorph

"""
Functional interface for morphological operations.
"""

__all__ = [
    'dilation', 'erosion', 'smorph', 'emorph'
]