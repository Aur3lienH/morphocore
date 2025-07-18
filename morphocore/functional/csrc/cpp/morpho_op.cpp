#include "morpho_op.hpp"




template torch::Tensor morpho_dilaton_forward_cpu<MORPHO_OPERATION::DILATION, float>(torch::Tensor input, torch::Tensor weight);
template torch::Tensor morpho_dilaton_forward_cpu<MORPHO_OPERATION::DILATION, double>(torch::Tensor input, torch::Tensor weight);
template torch::Tensor morpho_dilaton_forward_cpu<MORPHO_OPERATION::DILATION, torch::Half>(torch::Tensor input, torch::Tensor weight);
template torch::Tensor morpho_dilaton_forward_cpu<MORPHO_OPERATION::DILATION, torch::BFloat16>(torch::Tensor input, torch::Tensor weight);

template torch::autograd::tensor_list morpho_dilaton_backward_cpu<MORPHO_OPERATION::DILATION, float>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output);
template torch::autograd::tensor_list morpho_dilaton_backward_cpu<MORPHO_OPERATION::DILATION, double>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output);
template torch::autograd::tensor_list morpho_dilaton_backward_cpu<MORPHO_OPERATION::DILATION, torch::Half>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output);
template torch::autograd::tensor_list morpho_dilaton_backward_cpu<MORPHO_OPERATION::DILATION, torch::BFloat16>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output);


template torch::Tensor morpho_dilaton_forward_cpu<MORPHO_OPERATION::EROSION, float>(torch::Tensor input, torch::Tensor weight);
template torch::Tensor morpho_dilaton_forward_cpu<MORPHO_OPERATION::EROSION, double>(torch::Tensor input, torch::Tensor weight);
template torch::Tensor morpho_dilaton_forward_cpu<MORPHO_OPERATION::EROSION, torch::Half>(torch::Tensor input, torch::Tensor weight);
template torch::Tensor morpho_dilaton_forward_cpu<MORPHO_OPERATION::EROSION, torch::BFloat16>(torch::Tensor input, torch::Tensor weight);

template torch::autograd::tensor_list morpho_dilaton_backward_cpu<MORPHO_OPERATION::EROSION, float>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output);
template torch::autograd::tensor_list morpho_dilaton_backward_cpu<MORPHO_OPERATION::EROSION, double>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output);
template torch::autograd::tensor_list morpho_dilaton_backward_cpu<MORPHO_OPERATION::EROSION, torch::Half>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output);
template torch::autograd::tensor_list morpho_dilaton_backward_cpu<MORPHO_OPERATION::EROSION, torch::BFloat16>(torch::Tensor input,torch::Tensor weight,torch::Tensor grad_output);


template MorphoOpertion<MORPHO_OPERATION::DILATION>;
template MorphoOpertion<MORPHO_OPERATION::EROSION>;