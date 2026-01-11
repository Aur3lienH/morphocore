#pragma once
#include "morpho_op.cuh"
#include "../cpp/morpho_op.hpp"
#include "../cpp/morpho_op_enum.hpp"
#include "../cpp/morpho_merge_policy.hpp"

template <typename scalar_t, MORPHO_OPERATION Op>
struct MorphoPolicyGPU;

template <typename scalar_t>
struct MorphoPolicyGPU<scalar_t, MORPHO_OPERATION::DILATION>
{
    __device__ __forceinline__ static constexpr scalar_t initial_value() { return std::numeric_limits<scalar_t>::lowest(); }

    __device__ __forceinline__ static constexpr scalar_t combine(scalar_t input, scalar_t weight) { return input + weight; }

    __device__ __forceinline__ static constexpr bool is_better(scalar_t new_val, scalar_t current_best) { return new_val > current_best; }

    __device__ __forceinline__ static constexpr scalar_t padding_value() { return std::numeric_limits<scalar_t>::lowest(); }
};

template <typename scalar_t>
struct MorphoPolicyGPU<scalar_t, MORPHO_OPERATION::EROSION>
{
    __device__ __forceinline__ static constexpr scalar_t initial_value() { return std::numeric_limits<scalar_t>::max(); }

    __device__ __forceinline__ static constexpr scalar_t combine(scalar_t input, scalar_t weight) { return input - weight; }

    __device__ __forceinline__ static constexpr bool is_better(scalar_t new_val, scalar_t current_best) { return new_val < current_best; }

    __device__ __forceinline__ static constexpr scalar_t padding_value() { return std::numeric_limits<scalar_t>::max(); }
};