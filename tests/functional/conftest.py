import pytest
import torch


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return request.param


@pytest.fixture(params=['dilation', 'erosion', 'Sdilation', 'Serosion'])
def operation(request):
    return request.param