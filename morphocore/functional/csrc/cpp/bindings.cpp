#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "morpho_op.hpp"
#include "smorph.hpp"
#include "smorph_scm.hpp"

/// @brief Morphological dilation operation
/// @param input Input tensor
/// @param weight Structuring element
/// @param merge Merge channel mode
/// @return Output tensor
torch::Tensor morpho_dilation(torch::Tensor input, torch::Tensor weight, MORPHO_MERGE merge, MORPHO_PADDING padding, bool save_indices)
{
    switch (padding)
    {
        case MORPHO_PADDING::REPLICATE:
            switch (merge)
            {
                case MORPHO_MERGE::ADD:
                    return MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::ADD, MORPHO_PADDING::REPLICATE>::apply(input, weight,
                                                                                                                            save_indices);
                case MORPHO_MERGE::MAX:
                    return MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::MAX, MORPHO_PADDING::REPLICATE>::apply(input, weight,
                                                                                                                            save_indices);
                case MORPHO_MERGE::MIN:
                    return MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::MIN, MORPHO_PADDING::REPLICATE>::apply(input, weight,
                                                                                                                            save_indices);
                case MORPHO_MERGE::AVERAGE:
                    return MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::AVERAGE, MORPHO_PADDING::REPLICATE>::apply(input, weight,
                                                                                                                                save_indices);
                case MORPHO_MERGE::IDENTITY:
                    return MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::IDENTITY, MORPHO_PADDING::REPLICATE>::apply(input, weight,
                                                                                                                                 save_indices);
                default: TORCH_CHECK(false, "Unsupported merge operation for dilation"); return torch::Tensor();
            }
        case MORPHO_PADDING::GEODESIC:
            switch (merge)
            {
                case MORPHO_MERGE::ADD:
                    return MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::ADD, MORPHO_PADDING::GEODESIC>::apply(input, weight,
                                                                                                                           save_indices);
                case MORPHO_MERGE::MAX:
                    return MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::MAX, MORPHO_PADDING::GEODESIC>::apply(input, weight,
                                                                                                                           save_indices);
                case MORPHO_MERGE::MIN:
                    return MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::MIN, MORPHO_PADDING::GEODESIC>::apply(input, weight,
                                                                                                                           save_indices);
                case MORPHO_MERGE::AVERAGE:
                    return MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::AVERAGE, MORPHO_PADDING::GEODESIC>::apply(input, weight,
                                                                                                                               save_indices);
                case MORPHO_MERGE::IDENTITY:
                    return MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::IDENTITY, MORPHO_PADDING::GEODESIC>::apply(input, weight,
                                                                                                                                save_indices);
                default: TORCH_CHECK(false, "Unsupported merge operation for dilation"); return torch::Tensor();
            }
        default: TORCH_CHECK(false, "Unsupported padding mode for dilation"); return torch::Tensor();
    }
}

/// @brief Morphological erosion operation
/// @param input Input tensor
/// @param weight Structuring element
/// @param merge Merge channel mode
/// @return Output tensor
torch::Tensor morpho_erosion(torch::Tensor input, torch::Tensor weight, MORPHO_MERGE merge, MORPHO_PADDING padding, bool save_indices)
{
    switch (padding)
    {
        case MORPHO_PADDING::REPLICATE:
            switch (merge)
            {
                case MORPHO_MERGE::ADD:
                    return MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::ADD, MORPHO_PADDING::REPLICATE>::apply(input, weight,
                                                                                                                           save_indices);
                case MORPHO_MERGE::MAX:
                    return MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::MAX, MORPHO_PADDING::REPLICATE>::apply(input, weight,
                                                                                                                           save_indices);
                case MORPHO_MERGE::MIN:
                    return MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::MIN, MORPHO_PADDING::REPLICATE>::apply(input, weight,
                                                                                                                           save_indices);
                case MORPHO_MERGE::AVERAGE:
                    return MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::AVERAGE, MORPHO_PADDING::REPLICATE>::apply(input, weight,
                                                                                                                               save_indices);
                case MORPHO_MERGE::IDENTITY:
                    return MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::IDENTITY, MORPHO_PADDING::REPLICATE>::apply(input, weight,
                                                                                                                                save_indices);
                default: TORCH_CHECK(false, "Unsupported merge operation for erosion"); return torch::Tensor();
            }
        case MORPHO_PADDING::GEODESIC:
            switch (merge)
            {
                case MORPHO_MERGE::ADD:
                    return MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::ADD, MORPHO_PADDING::GEODESIC>::apply(input, weight,
                                                                                                                          save_indices);
                case MORPHO_MERGE::MAX:
                    return MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::MAX, MORPHO_PADDING::GEODESIC>::apply(input, weight,
                                                                                                                          save_indices);
                case MORPHO_MERGE::MIN:
                    return MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::MIN, MORPHO_PADDING::GEODESIC>::apply(input, weight,
                                                                                                                          save_indices);
                case MORPHO_MERGE::AVERAGE:
                    return MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::AVERAGE, MORPHO_PADDING::GEODESIC>::apply(input, weight,
                                                                                                                              save_indices);
                case MORPHO_MERGE::IDENTITY:
                    return MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::IDENTITY, MORPHO_PADDING::GEODESIC>::apply(input, weight,
                                                                                                                               save_indices);
                default: TORCH_CHECK(false, "Unsupported merge operation for erosion"); return torch::Tensor();
            }
        default: TORCH_CHECK(false, "Unsupported padding mode for erosion"); return torch::Tensor();
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
        case MORPHO_MERGE::IDENTITY: return Smorph<MORPHO_MERGE::IDENTITY>::apply(input, weight, alpha);
        default: TORCH_CHECK(false, "Unsupported merge operation for smorph"); return torch::Tensor();
    }
}

torch::Tensor smorph_scm(torch::Tensor input, torch::Tensor weight, torch::Tensor alpha, torch::Tensor beta)
{
    return SmorphSCM::apply(input, weight, alpha, beta);
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
        .value("IDENTITY", MORPHO_MERGE::IDENTITY)
        .export_values();

    py::enum_<MORPHO_PADDING>(m, "MorphoPadding")
        .value("GEODESIC", MORPHO_PADDING::GEODESIC)
        .value("REPLICATE", MORPHO_PADDING::REPLICATE)
        .export_values();

    m.def("morpho_dilation", &morpho_dilation, "Morphological dilation operation (CUDA)");
    m.def("morpho_erosion", &morpho_erosion, "Morphological erosion operation (CUDA)");
    m.def("smorph", &smorph, "Morphological approximation of erosion and dilation (CUDA)");
    m.def("smorph_scm", &smorph_scm, "Soft Morphological layer with Smooth Channel Merge (CUDA)");
}