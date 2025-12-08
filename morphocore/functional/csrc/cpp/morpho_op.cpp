#include "morpho_op.hpp"
#include "morpho_op_cpu_backward_linear.hxx"
#include "morpho_op_cpu_forward_linear.hxx"

// TODO: Add float16 support !
/// @Brief Template instantiations for morphological operations (dilation and erosion) with different merge policies (ADD, MAX, MIN, AVERAGE) and data
/// types (float, double).

// ----------------------------------TORCH CLASSES----------------------------------------

template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::ADD, MORPHO_PADDING::REPLICATE>;
template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::MAX, MORPHO_PADDING::REPLICATE>;
template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::MIN, MORPHO_PADDING::REPLICATE>;
template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::AVERAGE, MORPHO_PADDING::REPLICATE>;
template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::IDENTITY, MORPHO_PADDING::REPLICATE>;

template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::ADD, MORPHO_PADDING::GEODESIC>;
template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::MAX, MORPHO_PADDING::GEODESIC>;
template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::MIN, MORPHO_PADDING::GEODESIC>;
template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::AVERAGE, MORPHO_PADDING::GEODESIC>;
template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::IDENTITY, MORPHO_PADDING::GEODESIC>;

template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::ADD, MORPHO_PADDING::REPLICATE>;
template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::MAX, MORPHO_PADDING::REPLICATE>;
template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::MIN, MORPHO_PADDING::REPLICATE>;
template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::AVERAGE, MORPHO_PADDING::REPLICATE>;
template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::IDENTITY, MORPHO_PADDING::REPLICATE>;

template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::ADD, MORPHO_PADDING::GEODESIC>;
template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::MAX, MORPHO_PADDING::GEODESIC>;
template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::MIN, MORPHO_PADDING::GEODESIC>;
template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::AVERAGE, MORPHO_PADDING::GEODESIC>;
template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::IDENTITY, MORPHO_PADDING::GEODESIC>;

// ----------------------------------TORCH CLASSES----------------------------------------

// ----------------------------------FORWARD PASS CPU----------------------------------------

#define INSTANTIATE_FORWARD_ALL_MERGES(OP, PADDING, TYPE, SAVE_IDX)                                                                                  \
    template auto morpho_operation_forward_cpu<MORPHO_OPERATION::OP, MORPHO_MERGE::ADD, PADDING, TYPE, SAVE_IDX>(torch::Tensor, torch::Tensor);      \
    template auto morpho_operation_forward_cpu<MORPHO_OPERATION::OP, MORPHO_MERGE::MAX, PADDING, TYPE, SAVE_IDX>(torch::Tensor, torch::Tensor);      \
    template auto morpho_operation_forward_cpu<MORPHO_OPERATION::OP, MORPHO_MERGE::MIN, PADDING, TYPE, SAVE_IDX>(torch::Tensor, torch::Tensor);      \
    template auto morpho_operation_forward_cpu<MORPHO_OPERATION::OP, MORPHO_MERGE::AVERAGE, PADDING, TYPE, SAVE_IDX>(torch::Tensor, torch::Tensor);  \
    template auto morpho_operation_forward_cpu_dispatcher<MORPHO_OPERATION::OP, MORPHO_MERGE::ADD, PADDING, TYPE, SAVE_IDX>(torch::Tensor input,     \
                                                                                                                            torch::Tensor weight);   \
    template auto morpho_operation_forward_cpu_dispatcher<MORPHO_OPERATION::OP, MORPHO_MERGE::MAX, PADDING, TYPE, SAVE_IDX>(torch::Tensor input,     \
                                                                                                                            torch::Tensor weight);   \
    template auto morpho_operation_forward_cpu_dispatcher<MORPHO_OPERATION::OP, MORPHO_MERGE::MIN, PADDING, TYPE, SAVE_IDX>(torch::Tensor input,     \
                                                                                                                            torch::Tensor weight);   \
    template auto morpho_operation_forward_cpu_dispatcher<MORPHO_OPERATION::OP, MORPHO_MERGE::AVERAGE, PADDING, TYPE, SAVE_IDX>(                     \
        torch::Tensor input, torch::Tensor weight);

// FORWARD PASS:
// dilation or erosion
// float or double
// Save indices or not
INSTANTIATE_FORWARD_ALL_MERGES(DILATION, MORPHO_PADDING::REPLICATE, float, true)
INSTANTIATE_FORWARD_ALL_MERGES(DILATION, MORPHO_PADDING::REPLICATE, float, false)
INSTANTIATE_FORWARD_ALL_MERGES(DILATION, MORPHO_PADDING::REPLICATE, double, true)
INSTANTIATE_FORWARD_ALL_MERGES(DILATION, MORPHO_PADDING::REPLICATE, double, false)
INSTANTIATE_FORWARD_ALL_MERGES(EROSION, MORPHO_PADDING::REPLICATE, float, true)
INSTANTIATE_FORWARD_ALL_MERGES(EROSION, MORPHO_PADDING::REPLICATE, float, false)
INSTANTIATE_FORWARD_ALL_MERGES(EROSION, MORPHO_PADDING::REPLICATE, double, true);
INSTANTIATE_FORWARD_ALL_MERGES(EROSION, MORPHO_PADDING::REPLICATE, double, false);

INSTANTIATE_FORWARD_ALL_MERGES(DILATION, MORPHO_PADDING::GEODESIC, float, true)
INSTANTIATE_FORWARD_ALL_MERGES(DILATION, MORPHO_PADDING::GEODESIC, float, false)
INSTANTIATE_FORWARD_ALL_MERGES(DILATION, MORPHO_PADDING::GEODESIC, double, true)
INSTANTIATE_FORWARD_ALL_MERGES(DILATION, MORPHO_PADDING::GEODESIC, double, false)
INSTANTIATE_FORWARD_ALL_MERGES(EROSION, MORPHO_PADDING::GEODESIC, float, true)
INSTANTIATE_FORWARD_ALL_MERGES(EROSION, MORPHO_PADDING::GEODESIC, float, false)
INSTANTIATE_FORWARD_ALL_MERGES(EROSION, MORPHO_PADDING::GEODESIC, double, true);
INSTANTIATE_FORWARD_ALL_MERGES(EROSION, MORPHO_PADDING::GEODESIC, double, false);

template <MORPHO_OPERATION op, MORPHO_MERGE merge, MORPHO_PADDING padding, typename T, bool save_index>
auto morpho_operation_forward_cpu_dispatcher(torch::Tensor input, torch::Tensor weight);

// ----------------------------------FORWARD PASS CPU----------------------------------------

// ----------------------------------FORWARD SPECIALIZED PASS CPU----------------------------------------

#define INSTANTIATE_TEMPLATE_KERNELS(OP, PADDING, TYPE, SAVE_IDX)                                                                                    \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::ADD, PADDING, TYPE, SAVE_IDX, 3>(torch::Tensor,          \
                                                                                                                             torch::Tensor);         \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::MAX, PADDING, TYPE, SAVE_IDX, 3>(torch::Tensor,          \
                                                                                                                             torch::Tensor);         \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::MIN, PADDING, TYPE, SAVE_IDX, 3>(torch::Tensor,          \
                                                                                                                             torch::Tensor);         \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::AVERAGE, PADDING, TYPE, SAVE_IDX, 3>(torch::Tensor,      \
                                                                                                                                 torch::Tensor);     \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::ADD, PADDING, TYPE, SAVE_IDX, 5>(torch::Tensor,          \
                                                                                                                             torch::Tensor);         \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::MAX, PADDING, TYPE, SAVE_IDX, 5>(torch::Tensor,          \
                                                                                                                             torch::Tensor);         \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::MIN, PADDING, TYPE, SAVE_IDX, 5>(torch::Tensor,          \
                                                                                                                             torch::Tensor);         \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::AVERAGE, PADDING, TYPE, SAVE_IDX, 5>(torch::Tensor,      \
                                                                                                                                 torch::Tensor);     \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::ADD, PADDING, TYPE, SAVE_IDX, 7>(torch::Tensor,          \
                                                                                                                             torch::Tensor);         \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::MAX, PADDING, TYPE, SAVE_IDX, 7>(torch::Tensor,          \
                                                                                                                             torch::Tensor);         \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::MIN, PADDING, TYPE, SAVE_IDX, 7>(torch::Tensor,          \
                                                                                                                             torch::Tensor);         \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::AVERAGE, PADDING, TYPE, SAVE_IDX, 7>(torch::Tensor,      \
                                                                                                                                 torch::Tensor);

// INSTANTIATE_TEMPLATE_KERNELS(DILATION, MORPHO_PADDING::GEODESIC, float, true)
// INSTANTIATE_TEMPLATE_KERNELS(DILATION, MORPHO_PADDING::GEODESIC, float, false)
// INSTANTIATE_TEMPLATE_KERNELS(DILATION, MORPHO_PADDING::GEODESIC, double, true)
// INSTANTIATE_TEMPLATE_KERNELS(DILATION, MORPHO_PADDING::GEODESIC, double, false)
// INSTANTIATE_TEMPLATE_KERNELS(EROSION, MORPHO_PADDING::GEODESIC, float, true)
// INSTANTIATE_TEMPLATE_KERNELS(EROSION, MORPHO_PADDING::GEODESIC, float, false)
// INSTANTIATE_TEMPLATE_KERNELS(EROSION, MORPHO_PADDING::GEODESIC, double, true)
// INSTANTIATE_TEMPLATE_KERNELS(EROSION, MORPHO_PADDING::GEODESIC, double, false)

// INSTANTIATE_TEMPLATE_KERNELS(DILATION, MORPHO_PADDING::REPLICATE, float, true)
// INSTANTIATE_TEMPLATE_KERNELS(DILATION, MORPHO_PADDING::REPLICATE, float, false)
// INSTANTIATE_TEMPLATE_KERNELS(DILATION, MORPHO_PADDING::REPLICATE, double, true)
// INSTANTIATE_TEMPLATE_KERNELS(DILATION, MORPHO_PADDING::REPLICATE, double, false)
// INSTANTIATE_TEMPLATE_KERNELS(EROSION, MORPHO_PADDING::REPLICATE, float, true)
// INSTANTIATE_TEMPLATE_KERNELS(EROSION, MORPHO_PADDING::REPLICATE, float, false)
// INSTANTIATE_TEMPLATE_KERNELS(EROSION, MORPHO_PADDING::REPLICATE, double, true)
// INSTANTIATE_TEMPLATE_KERNELS(EROSION, MORPHO_PADDING::REPLICATE, double, false)

// ----------------------------------FORWARD SPECIALIZED PASS CPU----------------------------------------

// ----------------------------------BACKWARD PASS CPU----------------------------------------

#define INSTANTIATE_BACKWARD(OP, PADDING, TYPE, SAVE_IDX)                                                                                            \
    template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::OP, MORPHO_MERGE::ADD, PADDING, TYPE, SAVE_IDX>(           \
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);                                                                                 \
    template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::OP, MORPHO_MERGE::MAX, PADDING, TYPE, SAVE_IDX>(           \
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);                                                                                 \
    template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::OP, MORPHO_MERGE::MIN, PADDING, TYPE, SAVE_IDX>(           \
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);                                                                                 \
    template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::OP, MORPHO_MERGE::AVERAGE, PADDING, TYPE, SAVE_IDX>(       \
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);

INSTANTIATE_BACKWARD(DILATION, MORPHO_PADDING::REPLICATE, float, true)
INSTANTIATE_BACKWARD(DILATION, MORPHO_PADDING::REPLICATE, float, false)
INSTANTIATE_BACKWARD(DILATION, MORPHO_PADDING::REPLICATE, double, true)
INSTANTIATE_BACKWARD(DILATION, MORPHO_PADDING::REPLICATE, double, false)
INSTANTIATE_BACKWARD(EROSION, MORPHO_PADDING::REPLICATE, float, true)
INSTANTIATE_BACKWARD(EROSION, MORPHO_PADDING::REPLICATE, float, false)
INSTANTIATE_BACKWARD(EROSION, MORPHO_PADDING::REPLICATE, double, true)
INSTANTIATE_BACKWARD(EROSION, MORPHO_PADDING::REPLICATE, double, false)

INSTANTIATE_BACKWARD(DILATION, MORPHO_PADDING::GEODESIC, float, true)
INSTANTIATE_BACKWARD(DILATION, MORPHO_PADDING::GEODESIC, float, false)
INSTANTIATE_BACKWARD(DILATION, MORPHO_PADDING::GEODESIC, double, true)
INSTANTIATE_BACKWARD(DILATION, MORPHO_PADDING::GEODESIC, double, false)
INSTANTIATE_BACKWARD(EROSION, MORPHO_PADDING::GEODESIC, float, true)
INSTANTIATE_BACKWARD(EROSION, MORPHO_PADDING::GEODESIC, float, false)
INSTANTIATE_BACKWARD(EROSION, MORPHO_PADDING::GEODESIC, double, true)
INSTANTIATE_BACKWARD(EROSION, MORPHO_PADDING::GEODESIC, double, false)

// ----------------------------------BACKWARD PASS CPU----------------------------------------

#define INSTANTIATE_BACKWARD_IDENTITY(OP, PADDING, TYPE, SAVE_IDX)                                                                                   \
    template torch::autograd::tensor_list morpho_operation_backward_identity_cpu<MORPHO_OPERATION::OP, PADDING, TYPE, SAVE_IDX>(                     \
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);

INSTANTIATE_BACKWARD_IDENTITY(DILATION, MORPHO_PADDING::REPLICATE, float, true)
INSTANTIATE_BACKWARD_IDENTITY(DILATION, MORPHO_PADDING::REPLICATE, float, false)
INSTANTIATE_BACKWARD_IDENTITY(DILATION, MORPHO_PADDING::REPLICATE, double, true)
INSTANTIATE_BACKWARD_IDENTITY(DILATION, MORPHO_PADDING::REPLICATE, double, false)
INSTANTIATE_BACKWARD_IDENTITY(EROSION, MORPHO_PADDING::REPLICATE, float, true)
INSTANTIATE_BACKWARD_IDENTITY(EROSION, MORPHO_PADDING::REPLICATE, float, false)
INSTANTIATE_BACKWARD_IDENTITY(EROSION, MORPHO_PADDING::REPLICATE, double, true)
INSTANTIATE_BACKWARD_IDENTITY(EROSION, MORPHO_PADDING::REPLICATE, double, false)

INSTANTIATE_BACKWARD_IDENTITY(DILATION, MORPHO_PADDING::GEODESIC, float, true)
INSTANTIATE_BACKWARD_IDENTITY(DILATION, MORPHO_PADDING::GEODESIC, float, false)
INSTANTIATE_BACKWARD_IDENTITY(DILATION, MORPHO_PADDING::GEODESIC, double, true)
INSTANTIATE_BACKWARD_IDENTITY(DILATION, MORPHO_PADDING::GEODESIC, double, false)
INSTANTIATE_BACKWARD_IDENTITY(EROSION, MORPHO_PADDING::GEODESIC, float, true)
INSTANTIATE_BACKWARD_IDENTITY(EROSION, MORPHO_PADDING::GEODESIC, float, false)
INSTANTIATE_BACKWARD_IDENTITY(EROSION, MORPHO_PADDING::GEODESIC, double, true)
INSTANTIATE_BACKWARD_IDENTITY(EROSION, MORPHO_PADDING::GEODESIC, double, false)