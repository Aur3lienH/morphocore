#include "morpho_op.hpp"
#include "morpho_padding_policy.hpp"
#include "morpho_op_cpu_forward.hxx"
#include "morpho_op_cpu_backward.hxx"
#include "morpho_op_wrap.hxx"

// TODO: Add float16 support !
/// @Brief Template instantiations for morphological operations (dilation and erosion) with different merge policies (ADD, MAX, MIN, AVERAGE,
/// IDENTITY) and padding families (FILL, REPLICATE). Each class pulls in the CPU kernels it dispatches to, for float and double.

// ----------------------------------TORCH CLASSES----------------------------------------

template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::ADD, MORPHO_PADDING_KIND::REPLICATE>;
template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::MAX, MORPHO_PADDING_KIND::REPLICATE>;
template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::MIN, MORPHO_PADDING_KIND::REPLICATE>;
template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::AVERAGE, MORPHO_PADDING_KIND::REPLICATE>;
template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::IDENTITY, MORPHO_PADDING_KIND::REPLICATE>;

template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::ADD, MORPHO_PADDING_KIND::FILL>;
template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::MAX, MORPHO_PADDING_KIND::FILL>;
template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::MIN, MORPHO_PADDING_KIND::FILL>;
template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::AVERAGE, MORPHO_PADDING_KIND::FILL>;
template class MorphoOperation<MORPHO_OPERATION::DILATION, MORPHO_MERGE::IDENTITY, MORPHO_PADDING_KIND::FILL>;

template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::ADD, MORPHO_PADDING_KIND::REPLICATE>;
template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::MAX, MORPHO_PADDING_KIND::REPLICATE>;
template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::MIN, MORPHO_PADDING_KIND::REPLICATE>;
template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::AVERAGE, MORPHO_PADDING_KIND::REPLICATE>;
template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::IDENTITY, MORPHO_PADDING_KIND::REPLICATE>;

template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::ADD, MORPHO_PADDING_KIND::FILL>;
template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::MAX, MORPHO_PADDING_KIND::FILL>;
template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::MIN, MORPHO_PADDING_KIND::FILL>;
template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::AVERAGE, MORPHO_PADDING_KIND::FILL>;
template class MorphoOperation<MORPHO_OPERATION::EROSION, MORPHO_MERGE::IDENTITY, MORPHO_PADDING_KIND::FILL>;

// ----------------------------------TORCH CLASSES----------------------------------------
