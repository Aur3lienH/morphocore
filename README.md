# Morphocore

Morphocore is a Python library designed to integrate mathematical morphology into neural networks. It provides differentiable and optimized morphological operations, enabling deep learning architectures to leverage powerful image processing tools from mathematical morphology.

## Main Features
- Primary morphological operations (dilation, erosion)
- Creation of pytorch function (dilaiton )
- CPU and CUDA implementations for optimal performance
- Simple and modular API

## Installation

Installation from source: 

```bash
git clone https://github.com/your-username/morphocore.git
cd morphocore
pip install .
```

## Usage

### Basic Morphological Operations

```python
from morphocore.functional import dilation, erosion

result = dilation(image, structuring_element)
```

### Integration in Neural Networks (PyTorch)

```python
import torch
from morphocore.nn import Dilation

layer = Dilation(kernel_size=3)
output = layer(input_tensor)
```

## Project Structure
- `morphocore/functional/`: Basic morphological functions
- `morphocore/nn/`: PyTorch modules for neural networks
- `morphocore/functional/csrc/`: C++/CUDA source code for acceleration
- `tests/`: Unit tests

## Dependencies
- Python >= 3.8
- PyTorch
- NumPy

## Contributing
Contributions are welcome! Please submit issues and pull requests.

## License
This project is licensed under the MIT License.


