import pytest
import torch


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return request.param


@pytest.fixture(params=['Sdilation', 'Serosion', 'dilation', 'erosion'])
def operation(request):
    return request.param