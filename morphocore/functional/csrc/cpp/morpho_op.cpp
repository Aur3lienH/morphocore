#include "morpho_op.hpp"

//TODO: Add float16 support !
/// @Brief Template instantiations for morphological operations (dilation and erosion) with different merge policies (ADD, MAX, MIN, AVERAGE) and data types (float, double).

// ----------------------------------TORCH CLASSES----------------------------------------

template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::ADD>;
template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::MAX>;
template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::MIN>;
template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::AVERAGE>;

template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::ADD>;
template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::MAX>;
template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::MIN>;
template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::AVERAGE>;

// ----------------------------------TORCH CLASSES----------------------------------------

// ----------------------------------FORWARD PASS CPU----------------------------------------

#define INSTANTIATE_FORWARD_ALL_MERGES(OP, TYPE, SAVE_IDX)                                                                                           \
    template auto morpho_operation_forward_cpu<MORPHO_OPERATION::OP, MORPHO_MERGE::ADD, TYPE, SAVE_IDX>(torch::Tensor, torch::Tensor);               \
    template auto morpho_operation_forward_cpu<MORPHO_OPERATION::OP, MORPHO_MERGE::MAX, TYPE, SAVE_IDX>(torch::Tensor, torch::Tensor);               \
    template auto morpho_operation_forward_cpu<MORPHO_OPERATION::OP, MORPHO_MERGE::MIN, TYPE, SAVE_IDX>(torch::Tensor, torch::Tensor);               \
    template auto morpho_operation_forward_cpu<MORPHO_OPERATION::OP, MORPHO_MERGE::AVERAGE, TYPE, SAVE_IDX>(torch::Tensor, torch::Tensor);           \
    template auto morpho_operation_forward_cpu_dispatcher<MORPHO_OPERATION::OP, MORPHO_MERGE::ADD, TYPE, SAVE_IDX>(torch::Tensor input,              \
                                                                                                                   torch::Tensor weight);            \
    template auto morpho_operation_forward_cpu_dispatcher<MORPHO_OPERATION::OP, MORPHO_MERGE::MAX, TYPE, SAVE_IDX>(torch::Tensor input,              \
                                                                                                                   torch::Tensor weight);            \
    template auto morpho_operation_forward_cpu_dispatcher<MORPHO_OPERATION::OP, MORPHO_MERGE::MIN, TYPE, SAVE_IDX>(torch::Tensor input,              \
                                                                                                                   torch::Tensor weight);            \
    template auto morpho_operation_forward_cpu_dispatcher<MORPHO_OPERATION::OP, MORPHO_MERGE::AVERAGE, TYPE, SAVE_IDX>(torch::Tensor input,          \
                                                                                                                       torch::Tensor weight);

// FORWARD PASS:
// dilation or erosion
// float or double
// Save indices or not
INSTANTIATE_FORWARD_ALL_MERGES(DILATION, float, true)
INSTANTIATE_FORWARD_ALL_MERGES(DILATION, float, false)
INSTANTIATE_FORWARD_ALL_MERGES(DILATION, double, true)
INSTANTIATE_FORWARD_ALL_MERGES(DILATION, double, false)
INSTANTIATE_FORWARD_ALL_MERGES(EROSION, float, true)
INSTANTIATE_FORWARD_ALL_MERGES(EROSION, float, false)
INSTANTIATE_FORWARD_ALL_MERGES(EROSION, double, true);
INSTANTIATE_FORWARD_ALL_MERGES(EROSION, double, false);

template <MORPHO_OPERATION op, MORPHO_MERGE merge, typename T, bool save_index>
auto morpho_operation_forward_cpu_dispatcher(torch::Tensor input, torch::Tensor weight);

// ----------------------------------FORWARD PASS CPU----------------------------------------

// ----------------------------------FORWARD SPECIALIZED PASS CPU----------------------------------------

#define INSTANTIATE_TEMPLATE_KERNELS(OP, TYPE, SAVE_IDX)                                                                                             \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::ADD, TYPE, SAVE_IDX, 3>(torch::Tensor, torch::Tensor);   \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::MAX, TYPE, SAVE_IDX, 3>(torch::Tensor, torch::Tensor);   \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::MIN, TYPE, SAVE_IDX, 3>(torch::Tensor, torch::Tensor);   \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::AVERAGE, TYPE, SAVE_IDX, 3>(torch::Tensor,               \
                                                                                                                        torch::Tensor);              \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::ADD, TYPE, SAVE_IDX, 5>(torch::Tensor, torch::Tensor);   \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::MAX, TYPE, SAVE_IDX, 5>(torch::Tensor, torch::Tensor);   \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::MIN, TYPE, SAVE_IDX, 5>(torch::Tensor, torch::Tensor);   \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::AVERAGE, TYPE, SAVE_IDX, 5>(torch::Tensor,               \
                                                                                                                        torch::Tensor);              \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::ADD, TYPE, SAVE_IDX, 7>(torch::Tensor, torch::Tensor);   \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::MAX, TYPE, SAVE_IDX, 7>(torch::Tensor, torch::Tensor);   \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::MIN, TYPE, SAVE_IDX, 7>(torch::Tensor, torch::Tensor);   \
    template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::OP, MORPHO_MERGE::AVERAGE, TYPE, SAVE_IDX, 7>(torch::Tensor, torch::Tensor);

INSTANTIATE_TEMPLATE_KERNELS(DILATION, float, true)
INSTANTIATE_TEMPLATE_KERNELS(DILATION, float, false)
INSTANTIATE_TEMPLATE_KERNELS(DILATION, double, true)
INSTANTIATE_TEMPLATE_KERNELS(DILATION, double, false)
INSTANTIATE_TEMPLATE_KERNELS(EROSION, float, true)
INSTANTIATE_TEMPLATE_KERNELS(EROSION, float, false)
INSTANTIATE_TEMPLATE_KERNELS(EROSION, double, true)
INSTANTIATE_TEMPLATE_KERNELS(EROSION, double, false)

// ----------------------------------FORWARD SPECIALIZED PASS CPU----------------------------------------

// ----------------------------------BACKWARD PASS CPU----------------------------------------

#define INSTANTIATE_BACKWARD(OP, TYPE, SAVE_IDX)                                                                                                     \
    template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::OP, MORPHO_MERGE::ADD, TYPE, SAVE_IDX>(                    \
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);                                                                                 \
    template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::OP, MORPHO_MERGE::MAX, TYPE, SAVE_IDX>(                    \
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);                                                                                 \
    template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::OP, MORPHO_MERGE::MIN, TYPE, SAVE_IDX>(                    \
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);                                                                                 \
    template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::OP, MORPHO_MERGE::AVERAGE, TYPE, SAVE_IDX>(                \
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);

INSTANTIATE_BACKWARD(DILATION, float, true)
INSTANTIATE_BACKWARD(DILATION, float, false)
INSTANTIATE_BACKWARD(DILATION, double, true)
INSTANTIATE_BACKWARD(DILATION, double, false)
INSTANTIATE_BACKWARD(EROSION, float, true)
INSTANTIATE_BACKWARD(EROSION, float, false)
INSTANTIATE_BACKWARD(EROSION, double, true);
INSTANTIATE_BACKWARD(EROSION, double, false);

// ----------------------------------BACKWARD PASS CPU----------------------------------------
