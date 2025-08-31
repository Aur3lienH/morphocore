#include "morpho_op.hpp"




template auto morpho_operation_forward_cpu<MORPHO_OPERATION::DILATION, float, true>(torch::Tensor input, torch::Tensor weight);
template auto morpho_operation_forward_cpu<MORPHO_OPERATION::DILATION, double, true>(torch::Tensor input, torch::Tensor weight);
// template auto morpho_operation_forward_cpu<MORPHO_OPERATION::DILATION, torch::Half, true>(torch::Tensor input, torch::Tensor weight);
// template auto morpho_operation_forward_cpu<MORPHO_OPERATION::DILATION, torch::BFloat16, true>(torch::Tensor input, torch::Tensor weight);

template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::DILATION, float, true, 3>(torch::Tensor input, torch::Tensor weight);
template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::DILATION, double, true, 3>(torch::Tensor input, torch::Tensor weight); 
template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::DILATION, float, true, 5>(torch::Tensor input, torch::Tensor weight);
template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::DILATION, double, true, 5>(torch::Tensor input, torch::Tensor weight); 
template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::DILATION, float, true, 7>(torch::Tensor input, torch::Tensor weight);
template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::DILATION, double, true, 7>(torch::Tensor input, torch::Tensor weight); 

template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::EROSION, float, true, 3>(torch::Tensor input, torch::Tensor weight);
template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::EROSION, double, true, 3>(torch::Tensor input, torch::Tensor weight); 
template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::EROSION, float, true, 5>(torch::Tensor input, torch::Tensor weight);
template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::EROSION, double, true, 5>(torch::Tensor input, torch::Tensor weight); 
template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::EROSION, float, true, 7>(torch::Tensor input, torch::Tensor weight);
template auto morpho_operation_forward_cpu_template<MORPHO_OPERATION::EROSION, double, true, 7>(torch::Tensor input, torch::Tensor weight); 

template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::DILATION, float, true>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output, torch::Tensor max_indices);
template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::DILATION, double, true>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output, torch::Tensor max_indices);
// template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::DILATION, torch::Half, true>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output, torch::Tensor max_indices);
// template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::DILATION, torch::BFloat16, true>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output, torch::Tensor max_indices);

template auto morpho_operation_forward_cpu<MORPHO_OPERATION::DILATION, float, false>(torch::Tensor input, torch::Tensor weight);
template auto morpho_operation_forward_cpu<MORPHO_OPERATION::DILATION, double, false>(torch::Tensor input, torch::Tensor weight);
// template auto morpho_operation_forward_cpu<MORPHO_OPERATION::DILATION, torch::Half, false>(torch::Tensor input, torch::Tensor weight);
// template auto morpho_operation_forward_cpu<MORPHO_OPERATION::DILATION, torch::BFloat16, false>(torch::Tensor input, torch::Tensor weight);

template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::DILATION, float, false>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output, torch::Tensor max_indices);
template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::DILATION, double, false>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output, torch::Tensor max_indices);
// template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::DILATION, torch::Half, false>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output, torch::Tensor max_indices);
// template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::DILATION, torch::BFloat16, false>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output, torch::Tensor max_indices);


template auto morpho_operation_forward_cpu<MORPHO_OPERATION::EROSION, float, true>(torch::Tensor input, torch::Tensor weight);
template auto morpho_operation_forward_cpu<MORPHO_OPERATION::EROSION, double, true>(torch::Tensor input, torch::Tensor weight);
// template auto morpho_operation_forward_cpu<MORPHO_OPERATION::EROSION, torch::Half, true>(torch::Tensor input, torch::Tensor weight);
// template auto morpho_operation_forward_cpu<MORPHO_OPERATION::EROSION, torch::BFloat16, true>(torch::Tensor input, torch::Tensor weight);

template auto morpho_operation_forward_cpu<MORPHO_OPERATION::EROSION, float, false>(torch::Tensor input, torch::Tensor weight);
template auto morpho_operation_forward_cpu<MORPHO_OPERATION::EROSION, double, false>(torch::Tensor input, torch::Tensor weight);
// template auto morpho_operation_forward_cpu<MORPHO_OPERATION::EROSION, torch::Half, false>(torch::Tensor input, torch::Tensor weight);
// template auto morpho_operation_forward_cpu<MORPHO_OPERATION::EROSION, torch::BFloat16, false>(torch::Tensor input, torch::Tensor weight);

template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::EROSION, float, true>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output, torch::Tensor max_indices);
template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::EROSION, double, true>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output, torch::Tensor max_indices);
// template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::EROSION, torch::Half, true>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output, torch::Tensor max_indices);
// template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::EROSION, torch::BFloat16, true>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output, torch::Tensor max_indices);

template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::EROSION, float, false>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output, torch::Tensor max_indices);
template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::EROSION, double, false>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output, torch::Tensor max_indices);
// template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::EROSION, torch::Half, false>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output, torch::Tensor max_indices);
// template torch::autograd::tensor_list morpho_operation_backward_cpu<MORPHO_OPERATION::EROSION, torch::BFloat16, false>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output, torch::Tensor max_indices);


template class MorphoOperation<MORPHO_OPERATION::DILATION>;
template class MorphoOperation<MORPHO_OPERATION::EROSION>;