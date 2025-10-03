#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "morpho_op.hpp"
#include "smorph.hpp"

using MorphoDilationSum = MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::ADD>;
using MorphoDilationMax = MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::MAX>;
using MorphoDilationMin = MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::MIN>;
using MorphoDilationAverage = MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::AVERAGE>;

using MorphoErosionSum = MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::ADD>;
using MorphoErosionMax = MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::MAX>;
using MorphoErosionMin = MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::MIN>;
using MorphoErosionAverage = MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::AVERAGE>;

/// @brief Morphological dilation operation
/// @param input Input tensor
/// @param weight Structuring element
/// @param merge Merge channel mode
/// @return Output tensor
torch::Tensor morpho_dilation(torch::Tensor input, torch::Tensor weight, MORPHO_MERGE merge)
{
    switch (merge)
    {
        case MORPHO_MERGE::ADD: return MorphoDilationSum::apply(input, weight);
        case MORPHO_MERGE::MAX: return MorphoDilationMax::apply(input, weight);
        case MORPHO_MERGE::MIN: return MorphoDilationMin::apply(input, weight);
        case MORPHO_MERGE::AVERAGE: return MorphoDilationAverage::apply(input, weight);
        default: TORCH_CHECK(false, "Unsupported merge operation for dilation"); return torch::Tensor();
    }
}

/// @brief Morphological erosion operation
/// @param input Input tensor
/// @param weight Structuring element
/// @param merge Merge channel mode
/// @return Output tensor
torch::Tensor morpho_erosion(torch::Tensor input, torch::Tensor weight, MORPHO_MERGE merge)
{
    switch (merge)
    {
        case MORPHO_MERGE::ADD: return MorphoErosionSum::apply(input, weight);
        case MORPHO_MERGE::MAX: return MorphoErosionMax::apply(input, weight);
        case MORPHO_MERGE::MIN: return MorphoErosionMin::apply(input, weight);
        case MORPHO_MERGE::AVERAGE: return MorphoErosionAverage::apply(input, weight);
        default: TORCH_CHECK(false, "Unsupported merge operation for erosion"); return torch::Tensor();
    }
}

/// @brief Smorph operation (Smooth approximation of erosion or dilation) alpha->-inf => erosion, alpha > 0 => dilation
/// @param input Input tensor
/// @param weight Structuring element
/// @param alpha Smoothing factor
/// @param merge Merge channel mode
/// @return Output tensor
torch::Tensor smorph(torch::Tensor input, torch::Tensor weight, torch::Tensor alpha, MORPHO_MERGE merge)
{
    switch (merge)
    {
        case MORPHO_MERGE::ADD: return Smorph<MORPHO_MERGE::ADD>::apply(input, weight, alpha);
        case MORPHO_MERGE::MAX: return Smorph<MORPHO_MERGE::MAX>::apply(input, weight, alpha);
        case MORPHO_MERGE::MIN: return Smorph<MORPHO_MERGE::MIN>::apply(input, weight, alpha);
        case MORPHO_MERGE::AVERAGE: return Smorph<MORPHO_MERGE::AVERAGE>::apply(input, weight, alpha);
        default: TORCH_CHECK(false, "Unsupported merge operation for smorph"); return torch::Tensor();
    }
}

/// @brief Pybind11 module definition
PYBIND11_MODULE(cpp_operation, m)
{
    // Bind the MORPHO_MERGE enum for Python
    py::enum_<MORPHO_MERGE>(m, "MorphoMerge")
        .value("MAX", MORPHO_MERGE::MAX)
        .value("MIN", MORPHO_MERGE::MIN)
        .value("ADD", MORPHO_MERGE::ADD)
        .value("AVERAGE", MORPHO_MERGE::AVERAGE)
        .export_values();

    m.def("morpho_dilation", &morpho_dilation, "Morphological dilation operation (CUDA)");
    m.def("morpho_erosion", &morpho_erosion, "Morphological erosion operation (CUDA)");
    m.def("smorph", &smorph, "Morphological approximation of erosion and dilation (CUDA)");
}