#include "erosion.hpp"
#include "dilation.hpp"
#include "smorph.hpp"
#include <torch/torch.h>
#include <torch/extension.h>

torch::Tensor morpho_dilation(torch::Tensor input, torch::Tensor weight) {
    return MorphoDilation::apply(input, weight);
}

torch::Tensor morpho_erosion(torch::Tensor input, torch::Tensor weight) {
    return MorphoErosion::apply(input, weight);
}

torch::Tensor smorph(torch::Tensor input, torch::Tensor weight, torch::Tensor alpha){
    return Smorph::apply(input, weight, alpha);
}


PYBIND11_MODULE(cpp_operation, m) {
    m.def("morpho_dilation", &morpho_dilation, "Morphological dilation operation (CUDA)");
    m.def("morpho_erosion", &morpho_erosion, "Morphological erosion operation (CUDA)");
    m.def("smorph", &smorph, "Morphological approximation of erosion and dilation (CUDA)");
}