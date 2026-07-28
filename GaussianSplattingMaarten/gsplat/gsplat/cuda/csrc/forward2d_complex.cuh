#include "cuda_complex_types.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>


__global__ void project_complex_gaussians_2d_forward_kernel(
    const int num_points,
    const cuFloatComplex2* __restrict__ means2d, // Now complex 2D means using cuFloatComplex
    const cuFloatComplex3* __restrict__ L_elements, // Now complex L elements using cuFloatComplex
    const dim3 img_size,
    const dim3 tile_bounds,
    const float clip_thresh, // Unused in provided snippet, but kept for signature
    float2* __restrict__ xys, // Projected 2D screen coordinates (real)
    float* __restrict__ depths,
    int* __restrict__ radii,
    float3* __restrict__ conics, // Conic parameters (real)
    int32_t* __restrict__ num_tiles_hit
);
