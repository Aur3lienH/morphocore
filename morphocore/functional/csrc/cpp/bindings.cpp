#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "morpho_op.hpp"
#include "smorph.hpp"
#include "smorph_scm.hpp"

/// @brief Turn the runtime merge mode into a template argument, the operation and padding family
///        being already fixed by the caller.
/// @tparam op Morphological operation (dilation or erosion).
/// @tparam kind Padding family that implements `padding`.
template <MORPHO_OPERATION op, MORPHO_PADDING_KIND kind>
static torch::Tensor morpho_apply_merge(torch::Tensor input, torch::Tensor weight, MORPHO_MERGE merge, MORPHO_PADDING padding, double padding_value,
                                        bool save_indices)
{
    switch (merge)
    {
        case MORPHO_MERGE::ADD:
            return MorphoOperation<op, MORPHO_MERGE::ADD, kind>::apply(input, weight, save_indices, padding, padding_value);
        case MORPHO_MERGE::MAX:
            return MorphoOperation<op, MORPHO_MERGE::MAX, kind>::apply(input, weight, save_indices, padding, padding_value);
        case MORPHO_MERGE::MIN:
            return MorphoOperation<op, MORPHO_MERGE::MIN, kind>::apply(input, weight, save_indices, padding, padding_value);
        case MORPHO_MERGE::AVERAGE:
            return MorphoOperation<op, MORPHO_MERGE::AVERAGE, kind>::apply(input, weight, save_indices, padding, padding_value);
        case MORPHO_MERGE::IDENTITY:
            return MorphoOperation<op, MORPHO_MERGE::IDENTITY, kind>::apply(input, weight, save_indices, padding, padding_value);
        default: TORCH_CHECK(false, "Unsupported merge operation"); return torch::Tensor();
    }
}

/// @brief Entry point shared by dilation and erosion.
/// Only the padding *family* reaches the templates: zero, geodesic and constant all resolve to the
/// same instantiation and differ by the runtime padding_value alone. A new value based mode is a
/// case in morpho_fill_value, and costs nothing here.
/// @param padding Padding mode.
/// @param padding_value Value used by MORPHO_PADDING::CONSTANT, ignored by the other modes.
template <MORPHO_OPERATION op>
static torch::Tensor morpho_apply(torch::Tensor input, torch::Tensor weight, MORPHO_MERGE merge, MORPHO_PADDING padding, double padding_value,
                                  bool save_indices)
{
    switch (padding_kind(padding))
    {
        case MORPHO_PADDING_KIND::REPLICATE:
            return morpho_apply_merge<op, MORPHO_PADDING_KIND::REPLICATE>(input, weight, merge, padding, padding_value, save_indices);
        case MORPHO_PADDING_KIND::FILL:
            return morpho_apply_merge<op, MORPHO_PADDING_KIND::FILL>(input, weight, merge, padding, padding_value, save_indices);
        default: TORCH_CHECK(false, "Unsupported padding mode"); return torch::Tensor();
    }
}

/// @brief Morphological dilation operation
/// @param input Input tensor
/// @param weight Structuring element
/// @param merge Merge channel mode
/// @param padding Padding mode used on the borders
/// @param padding_value Value used by MORPHO_PADDING::CONSTANT, ignored by the other modes
/// @return Output tensor
torch::Tensor morpho_dilation(torch::Tensor input, torch::Tensor weight, MORPHO_MERGE merge, MORPHO_PADDING padding, double padding_value,
                              bool save_indices)
{
    return morpho_apply<MORPHO_OPERATION::DILATION>(input, weight, merge, padding, padding_value, save_indices);
}

/// @brief Morphological erosion operation
/// @param input Input tensor
/// @param weight Structuring element
/// @param merge Merge channel mode
/// @param padding Padding mode used on the borders
/// @param padding_value Value used by MORPHO_PADDING::CONSTANT, ignored by the other modes
/// @return Output tensor
torch::Tensor morpho_erosion(torch::Tensor input, torch::Tensor weight, MORPHO_MERGE merge, MORPHO_PADDING padding, double padding_value,
                             bool save_indices)
{
    return morpho_apply<MORPHO_OPERATION::EROSION>(input, weight, merge, padding, padding_value, save_indices);
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
        .value("ZERO", MORPHO_PADDING::ZERO)
        .value("GEODESIC", MORPHO_PADDING::GEODESIC)
        .value("CONSTANT", MORPHO_PADDING::CONSTANT)
        .value("REPLICATE", MORPHO_PADDING::REPLICATE)
        .export_values();

    m.def("morpho_dilation", &morpho_dilation, "Morphological dilation operation (CUDA)", py::arg("input"), py::arg("weight"), py::arg("merge"),
          py::arg("padding"), py::arg("padding_value") = 0.0, py::arg("save_indices") = true);
    m.def("morpho_erosion", &morpho_erosion, "Morphological erosion operation (CUDA)", py::arg("input"), py::arg("weight"), py::arg("merge"),
          py::arg("padding"), py::arg("padding_value") = 0.0, py::arg("save_indices") = true);
    m.def("smorph", &smorph, "Morphological approximation of erosion and dilation (CUDA)");
    m.def("smorph_scm", &smorph_scm, "Soft Morphological layer with Smooth Channel Merge (CUDA)");
}