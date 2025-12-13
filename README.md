# Morphocore

Morph### Installation from Source

```bash
git clone git@gitlab.lre.epita.fr:aurelien.hurand/morphocore.git
cd morphocore
pip install . --no-build-isolation
```

**CPU-Only Installation** (without CUDA):

```bash
MORPHO_WITH_CUDA=0 pip install . --no-build-isolation
```

This is useful when:
- You don't have an NVIDIA GPU
- CUDA Toolkit is not installed
- You want faster compilation times
- You're deploying on CPU-only serverss a Python library designed to integrate mathematical morphology into neural networks. It provides differentiable (float and double) and optimized morphological operations, enabling deep learning architectures to leverage powerful image processing tools from mathematical morphology.

## Main Features
- Primary morphological operations (dilation, erosion)
- Smorph (An alpha softmax based approximation of dilation or erosion : https://www.lre.epita.fr/dload/papers/hermary.22.jmiv.pdf)
- CPU and CUDA implementations (Related work: https://github.com/Manza12/nnMorpho)

## Installation

### Installation from PyPi:

```bash
pip install morphocore
```

❗If you have compatibility issue with the version compile on Pypi please compile the library from source like below !

### Installation from source: 

```bash
git clone git@gitlab.lre.epita.fr:aurelien.hurand/morphocore.git
cd morphocore
pip install . --no-build-isolation
```

### Requirements for Pre-compiled Package (PyPI)

If you install from PyPI using `pip install morphocore`, you'll need:

- Python >= 3.10
- PyTorch >= 2.6
- CUDA >= 12.3

If your environment doesn't meet these requirements, please compile from source instead.

## API Reference

### Functional Interface

#### `dilation(input, weight, channel_merge_mode="max")`
Performs morphological dilation on the input tensor.

**Parameters:**
- `input` (Tensor): Input tensor of shape (B, C, H, W)
- `weight` (Tensor): Structuring element of shape (out_channels, in_channels, kH, kW)
- `channel_merge_mode` (str): Channel combination method - "max", "min", "sum", "mean", or "identity"

**Returns:**
- `output` (Tensor): Dilated tensor of shape (B, out_channels, H, W)

```python
from morphocore.functional import dilation

# Basic usage
result = dilation(image, structuring_element, "sum")

# With different merge modes
max_result = dilation(image, kernel, "max")    # Maximum across channels
sum_result = dilation(image, kernel, "sum")    # Sum across channels
mean_result = dilation(image, kernel, "mean")  # Average across channels
identity_result = dilation(image, kernel, "identity")  # No channel merging, keeping all channels linear in_channels = out_channels
```

#### `erosion(input, weight, channel_merge_mode="max")`
Performs morphological erosion on the input tensor.

**Parameters:** Same as `dilation()`
**Returns:** Same as `dilation()`

```python
from morphocore.functional import erosion

result = erosion(image, structuring_element, "sum")
```

#### `smorph(input, weight, channel_merge_mode="max", alpha=0.0)`
Smooth approximation of morphological operations using softmax.

**Parameters:**
- Same as `dilation()` plus:
- `alpha` (float): Control parameter for either dilation or erosion behaviour
    - When alpha is large -> smorph behaves like dilation
    - When alpha is very negative -> smorph behaves like erosion
    - When alpha is close to 0 -> then smorph is something between an erosion and a dilation.

```python
from morphocore.functional import smorph

# Soft approximation of morphological operations
result = smorph(image, kernel, "sum", alpha=0.0)
```

### Neural Network Modules

#### `Mnn.Dilation(in_channels, out_channels, kernel_size, channel_merge_mode="max")`
Learnable dilation layer for neural networks.

**Parameters:**
- `in_channels` (int): Number of input channels
- `out_channels` (int): Number of output channels  
- `kernel_size` (int or tuple): Size of the morphological kernel
- `channel_merge_mode` (str): Channel merge strategy

```python
import morphocore.nn as Mnn

# Create a learnable dilation layer
dilation_layer = Mnn.Dilation(
    in_channels=3, 
    out_channels=16, 
    kernel_size=(3, 3), 
    channel_merge_mode="sum"
)
output = dilation_layer(input_tensor)
```

#### `Mnn.Erosion(in_channels, out_channels, kernel_size, channel_merge_mode="max")`
Learnable erosion layer for neural networks.

```python
erosion_layer = Mnn.Erosion(
    in_channels=3, 
    out_channels=16, 
    kernel_size=(5, 5), 
    channel_merge_mode="mean"
)
output = erosion_layer(input_tensor)
```

#### `Mnn.SMorph(in_channels, out_channels, kernel_size, channel_merge_mode="max", alpha=1.0)`
Smooth morphological layer using softmax approximation.

```python
smorph_layer = Mnn.SMorph(
    in_channels=3, 
    out_channels=8, 
    kernel_size=(3, 3), 
    channel_merge_mode="sum",
    alpha=2.0
)
output = smorph_layer(input_tensor)
```

## Channel Merge Modes Explained

- **"sum"**: Sums values across input channels
- **"mean"**: Averages values across input channels

## Project Structure
- `morphocore/functional/`: Basic morphological functions
- `morphocore/nn/`: PyTorch modules for neural networks
- `morphocore/functional/csrc/`: C++/CUDA source code for acceleration
- `tests/`: Unit, functional and benchmark tests

## Dependencies
- Python >= 3.8
- PyTorch
- NumPy

## Contributing
Contributions are welcome! Please submit issues and pull requests.

## License
This project is licensed under the MIT License.