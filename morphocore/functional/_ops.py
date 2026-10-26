import torch

# Loading the extension registers its operators with torch, under torch.ops.morphocore.
import morphocore._C  # noqa: F401

ops = torch.ops.morphocore
