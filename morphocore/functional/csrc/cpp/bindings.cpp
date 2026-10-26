// Python's stable ABI: the module below only needs PyModule_Create, so that one wheel serves every
// Python version. torch >= 2.9 defines Py_LIMITED_API itself for py_limited_api extensions.
#ifndef Py_LIMITED_API
    #define Py_LIMITED_API 0x030B0000
#endif
#include <Python.h>

#include <torch/all.h>
#include <torch/library.h>

#include "morpho_op.hpp"
#include "smorph.hpp"
#include "smorph_scm.hpp"

/// @brief Check an int coming from Python and turn it into a merge mode.
static MORPHO_MERGE to_merge(int64_t merge)
{
    TORCH_CHECK(merge >= 0 && merge <= static_cast<int64_t>(MORPHO_MERGE::IDENTITY), "Unknown merge mode ", merge);
    return static_cast<MORPHO_MERGE>(merge);
}

/// @brief Check an int coming from Python and turn it into a padding mode.
static MORPHO_PADDING to_padding(int64_t padding)
{
    TORCH_CHECK(padding >= 0 && padding <= static_cast<int64_t>(MORPHO_PADDING::REPLICATE), "Unknown padding mode ", padding);
    return static_cast<MORPHO_PADDING>(padding);
}

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
        case MORPHO_MERGE::ADD: return MorphoOperation<op, MORPHO_MERGE::ADD, kind>::apply(input, weight, save_indices, padding, padding_value);
        case MORPHO_MERGE::MAX: return MorphoOperation<op, MORPHO_MERGE::MAX, kind>::apply(input, weight, save_indices, padding, padding_value);
        case MORPHO_MERGE::MIN: return MorphoOperation<op, MORPHO_MERGE::MIN, kind>::apply(input, weight, save_indices, padding, padding_value);
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
static torch::Tensor morpho_apply(torch::Tensor input, torch::Tensor weight, int64_t merge, int64_t padding, double padding_value, bool save_indices)
{
    const MORPHO_PADDING mode = to_padding(padding);
    switch (padding_kind(mode))
    {
        case MORPHO_PADDING_KIND::REPLICATE:
            return morpho_apply_merge<op, MORPHO_PADDING_KIND::REPLICATE>(input, weight, to_merge(merge), mode, padding_value, save_indices);
        case MORPHO_PADDING_KIND::FILL:
            return morpho_apply_merge<op, MORPHO_PADDING_KIND::FILL>(input, weight, to_merge(merge), mode, padding_value, save_indices);
        default: TORCH_CHECK(false, "Unsupported padding mode"); return torch::Tensor();
    }
}

/// @brief Morphological dilation operation
/// @param input Input tensor
/// @param weight Structuring element
/// @param merge Merge channel mode, a MORPHO_MERGE value
/// @param padding Padding mode used on the borders, a MORPHO_PADDING value
/// @param padding_value Value used by MORPHO_PADDING::CONSTANT, ignored by the other modes
/// @return Output tensor
static torch::Tensor morpho_dilation(torch::Tensor input, torch::Tensor weight, int64_t merge, int64_t padding, double padding_value,
                                     bool save_indices)
{
    return morpho_apply<MORPHO_OPERATION::DILATION>(input, weight, merge, padding, padding_value, save_indices);
}

/// @brief Morphological erosion operation
/// @param input Input tensor
/// @param weight Structuring element
/// @param merge Merge channel mode, a MORPHO_MERGE value
/// @param padding Padding mode used on the borders, a MORPHO_PADDING value
/// @param padding_value Value used by MORPHO_PADDING::CONSTANT, ignored by the other modes
/// @return Output tensor
static torch::Tensor morpho_erosion(torch::Tensor input, torch::Tensor weight, int64_t merge, int64_t padding, double padding_value,
                                    bool save_indices)
{
    return morpho_apply<MORPHO_OPERATION::EROSION>(input, weight, merge, padding, padding_value, save_indices);
}

/// @brief Smorph operation (Smooth approximation of erosion or dilation) alpha->-inf => erosion, alpha > 0 => dilation
/// @param input Input tensor
/// @param weight Structuring element
/// @param alpha Smoothing factor
/// @param merge Merge channel mode, a MORPHO_MERGE value
/// @return Output tensor
static torch::Tensor smorph(torch::Tensor input, torch::Tensor weight, torch::Tensor alpha, int64_t merge)
{
    switch (to_merge(merge))
    {
        case MORPHO_MERGE::ADD: return Smorph<MORPHO_MERGE::ADD>::apply(input, weight, alpha);
        case MORPHO_MERGE::MAX: return Smorph<MORPHO_MERGE::MAX>::apply(input, weight, alpha);
        case MORPHO_MERGE::MIN: return Smorph<MORPHO_MERGE::MIN>::apply(input, weight, alpha);
        case MORPHO_MERGE::AVERAGE: return Smorph<MORPHO_MERGE::AVERAGE>::apply(input, weight, alpha);
        case MORPHO_MERGE::IDENTITY: return Smorph<MORPHO_MERGE::IDENTITY>::apply(input, weight, alpha);
        default: TORCH_CHECK(false, "Unsupported merge operation for smorph"); return torch::Tensor();
    }
}

static torch::Tensor smorph_scm(torch::Tensor input, torch::Tensor weight, torch::Tensor alpha, torch::Tensor beta)
{
    return SmorphSCM::apply(input, weight, alpha, beta);
}

// The operators are registered with torch's dispatcher instead of pybind11, which ties a module to one
// Python version, and are called from Python as torch.ops.morphocore.<name>. The merge and padding
// modes are the int values of MORPHO_MERGE and MORPHO_PADDING, mirrored by merge_enum.py.
TORCH_LIBRARY(morphocore, m)
{
    m.def("dilation(Tensor input, Tensor weight, int merge, int padding, float padding_value=0., bool save_indices=True) "
          "-> Tensor");
    m.def("erosion(Tensor input, Tensor weight, int merge, int padding, float padding_value=0., bool save_indices=True) "
          "-> Tensor");
    m.def("smorph(Tensor input, Tensor weight, Tensor alpha, int merge) -> Tensor");
    m.def("smorph_scm(Tensor input, Tensor weight, Tensor alpha, Tensor beta) -> Tensor");
}

// CompositeImplicitAutograd: the kernels run above autograd, for every device, and the autograd
// Functions they call record the backward pass themselves.
TORCH_LIBRARY_IMPL(morphocore, CompositeImplicitAutograd, m)
{
    m.impl("dilation", &morpho_dilation);
    m.impl("erosion", &morpho_erosion);
    m.impl("smorph", &smorph);
    m.impl("smorph_scm", &smorph_scm);
}

// Importing morphocore._C loads this library, which runs the registrations above. The module itself
// is empty. PyMODINIT_FUNC exports it, which a Windows DLL needs explicitly.
PyMODINIT_FUNC PyInit__C(void)
{
    static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr};
    return PyModule_Create(&module_def);
}
