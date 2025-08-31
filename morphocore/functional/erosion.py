from cpp_operation import morpho_erosion
import torch.functional

def erosion(x, w):

    """
    Make an erosion by calling C++ or CUDA.

    Args:
        x (torch.Tensor): input
        w (torch.Tensor): weight
    """
    return morpho_erosion(x, w)