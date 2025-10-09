#pragma once
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "morpho_merge_policy.cuh"
#include "morpho_op_policy.cuh"
#include "../cpp/morpho_op_enum.hpp"

/// @brief Kernel backward complet pour smorph avec merge policy ADD
/// Utilise shared memory pour optimiser les accès mémoire
template <MORPHO_MERGE merge, typename scalar_t>
__global__ void smorph_backward_kernel(const scalar_t* grad_output,  // [batch, out_channels, height, width]
                                       scalar_t* grad_input,         // [batch, in_channels, height, width]
                                       scalar_t* grad_weight,        // [out_channels, in_channels, kernel_h, kernel_w]
                                       scalar_t* grad_alpha,         // [1]
                                       const scalar_t* alpha,        // [1]
                                       const scalar_t* input,        // [batch, in_channels, height, width]
                                       const scalar_t* weights,      // [out_channels, in_channels, kernel_h, kernel_w]
                                       int batch_size, int input_channels, int output_channels, int height, int width, int kernel_height,
                                       int kernel_width, int kernel_height_half, int kernel_width_half)
{
    // ==================== SETUP SHARED MEMORY ====================
    extern __shared__ char shared_mem[];

    const int total_weight_size = output_channels * input_channels * kernel_height * kernel_width;
    const int PADDED_TILE_WIDTH = TILE_WIDTH + 2 * kernel_width_half;
    const int PADDED_TILE_HEIGHT = TILE_HEIGHT + 2 * kernel_height_half;
    const int padded_tile_size = PADDED_TILE_HEIGHT * PADDED_TILE_WIDTH;

    // Organisation de la shared memory
    scalar_t* shared_weight = (scalar_t*)shared_mem;
    scalar_t* shared_input = shared_weight + total_weight_size;
    // shared_input a besoin de: input_channels * padded_tile_size

    // ==================== CALCUL DES INDICES ====================
    const int b = blockIdx.z;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = blockIdx.x * blockDim.x + threadIdx.x;

    const int local_h = threadIdx.y;
    const int local_w = threadIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads_per_block = blockDim.x * blockDim.y;

    // ==================== CHARGEMENT DES POIDS EN SHARED MEMORY ====================
    for (int idx = tid; idx < total_weight_size; idx += threads_per_block)
    {
        shared_weight[idx] = weights[idx];
    }

    // ==================== CHARGEMENT DE L'INPUT EN SHARED MEMORY ====================
    const int tile_start_x = blockIdx.x * TILE_WIDTH - kernel_width_half;
    const int tile_start_y = blockIdx.y * TILE_HEIGHT - kernel_height_half;

    for (int ci = 0; ci < input_channels; ci++)
    {
        for (int i = tid; i < PADDED_TILE_HEIGHT * PADDED_TILE_WIDTH; i += threads_per_block)
        {
            int local_y = i / PADDED_TILE_WIDTH;
            int local_x = i % PADDED_TILE_WIDTH;

            int global_h = tile_start_y + local_y;
            int global_w = tile_start_x + local_x;

            int shared_idx = ci * padded_tile_size + local_y * PADDED_TILE_WIDTH + local_x;

            // Vérifier si la position est valide
            if (global_h >= 0 && global_h < height && global_w >= 0 && global_w < width)
            {
                int global_idx = b * input_channels * height * width + ci * height * width + global_h * width + global_w;
                shared_input[shared_idx] = input[global_idx];
            }
            else
            {
                // Marquer les positions hors limites avec NaN
                shared_input[shared_idx] = nanf("");
            }
        }
    }

    __syncthreads();

    // ==================== VÉRIFICATION DES BORNES ====================
    if (h >= height || w >= width || b >= batch_size) return;

    // ==================== RÉCUPÉRATION DE ALPHA ====================
    const scalar_t alpha_val = alpha[0];
    const scalar_t abs_alpha = abs(alpha_val);
    const bool is_dilation_like = alpha_val >= 0;

    // ==================== BOUCLE SUR LES CANAUX DE SORTIE ====================
    for (int co = 0; co < output_channels; co++)
    {
        // Récupérer le gradient de sortie pour cette position
        int grad_out_idx = b * output_channels * height * width + co * height * width + h * width + w;
        scalar_t grad_out = grad_output[grad_out_idx];

        if (grad_out == static_cast<scalar_t>(0)) continue;

        // Pour ADD merge: pas de scaling (pour AVERAGE ce serait 1/input_channels)
        const scalar_t scaled_grad = grad_out;

        // ==================== BOUCLE SUR LES CANAUX D'ENTRÉE ====================
        for (int ci = 0; ci < input_channels; ci++)
        {
            // Arrays pour stocker les calculs forward (sur la stack/registres)
            scalar_t vals[256];  // Assume kernel_height * kernel_width <= 256
            bool valid_pos[256];

            const int kernel_size = kernel_height * kernel_width;

            // ==================== PASSE 1: CALCULER VALS ET REFERENCE_VALUE ====================
            scalar_t reference_value = is_dilation_like ? -INFINITY : INFINITY;

            int k = 0;
            for (int kh = 0; kh < kernel_height; kh++)
            {
                for (int kw = 0; kw < kernel_width; kw++, k++)
                {
                    // Position dans la shared memory (avec padding)
                    // Match forward logic: ph = h + kernel_height_half - kh
                    int shared_h = local_h + kernel_height_half + kernel_height_half - kh;
                    int shared_w = local_w + kernel_width_half + kernel_width_half - kw;

                    if (shared_h >= 0 && shared_h < PADDED_TILE_HEIGHT && shared_w >= 0 && shared_w < PADDED_TILE_WIDTH)
                    {

                        int shared_input_idx = ci * padded_tile_size + shared_h * PADDED_TILE_WIDTH + shared_w;

                        scalar_t input_val = shared_input[shared_input_idx];

                        // Vérifier si la position est valide (pas NaN)
                        if (!isnan(input_val))
                        {
                            int weight_idx =
                                co * input_channels * kernel_height * kernel_width + ci * kernel_height * kernel_width + kh * kernel_width + kw;

                            scalar_t weight_val = shared_weight[weight_idx];
                            vals[k] = input_val + weight_val;
                            valid_pos[k] = true;

                            // Mettre à jour reference_value
                            if (is_dilation_like)
                            {
                                reference_value = max(vals[k], reference_value);
                            }
                            else
                            {
                                reference_value = min(vals[k], reference_value);
                            }
                        }
                        else
                        {
                            valid_pos[k] = false;
                        }
                    }
                    else
                    {
                        valid_pos[k] = false;
                    }
                }
            }

            // ==================== PASSE 2: CALCULER SOFT_WEIGHTS ====================
            scalar_t soft_weights[256];
            scalar_t sum = static_cast<scalar_t>(0);

            for (k = 0; k < kernel_size; k++)
            {
                if (valid_pos[k])
                {
                    scalar_t exp_term =
                        is_dilation_like ? __expf((vals[k] - reference_value) * abs_alpha) : __expf((reference_value - vals[k]) * abs_alpha);
                    soft_weights[k] = exp_term;
                    sum += exp_term;
                }
                else
                {
                    soft_weights[k] = static_cast<scalar_t>(0);
                }
            }

            // Normaliser les soft_weights
            if (sum > static_cast<scalar_t>(0))
            {
                for (k = 0; k < kernel_size; k++)
                {
                    soft_weights[k] /= sum;
                }
            }

            // ==================== PASSE 3: CALCULER WEIGHTED_RESULT (y_i) ====================
            scalar_t weighted_result = static_cast<scalar_t>(0);
            for (k = 0; k < kernel_size; k++)
            {
                if (valid_pos[k])
                {
                    weighted_result += soft_weights[k] * vals[k];
                }
            }

            // ==================== PASSE 4: CALCULER ET ACCUMULER LES GRADIENTS ====================
            k = 0;
            for (int kh = 0; kh < kernel_height; kh++)
            {
                for (int kw = 0; kw < kernel_width; kw++, k++)
                {
                    if (!valid_pos[k]) continue;

                    scalar_t p = soft_weights[k];
                    scalar_t val = vals[k];

                    // Formule du gradient: ∂y_i/∂v_k = p_k * [1 + α * (v_k - y_i)]
                    scalar_t grad_val = p + alpha_val * p * (val - weighted_result);
                    scalar_t final_grad = grad_val * scaled_grad;

                    // ======================== CORRECTED INDEXING ========================
                    // Calculate the GLOBAL input position that contributed to output (h, w)
                    // This matches forward pass logic: input_pos = h + kernel_height_half - kh
                    int global_h_target = h + kernel_height_half - kh;
                    int global_w_target = w + kernel_width_half - kw;
                    // ====================================================================

                    if (global_h_target >= 0 && global_h_target < height && global_w_target >= 0 && global_w_target < width)
                    {

                        // Accumuler gradient pour l'input
                        int input_idx = b * input_channels * height * width + ci * height * width + global_h_target * width + global_w_target;
                        atomicAdd(&grad_input[input_idx], final_grad);

                        // Accumuler gradient pour les poids
                        int weight_idx =
                            co * input_channels * kernel_height * kernel_width + ci * kernel_height * kernel_width + kh * kernel_width + kw;
                        atomicAdd(&grad_weight[weight_idx], final_grad);

                        // Accumuler gradient pour alpha
                        atomicAdd(grad_alpha, p * (val - reference_value) * (val - weighted_result) * scaled_grad);
                    }
                }
            }
        }
    }
}