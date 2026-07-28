#include "cuda_complex_types.h"
#include "forward2d.cuh"
#include "helpers.cuh"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuComplex.h> // Include CUDA's native complex number header
#include <iostream>
namespace cg = cooperative_groups;


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
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }

    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    // Retrieve the 2D Gaussian complex mean.
    // We use the real part for projection to screen coordinates.
    // The principal point (cx, cy) for ndc2pix is typically img_size.x * 0.5f, img_size.y * 0.5f
    // given the original calculation: 0.5f * W * x + 0.5f * W
    float2 center = {
        ndc2pix(cuCrealf(means2d[idx].x), img_size.x, 0.5f * img_size.x),
        ndc2pix(cuCrealf(means2d[idx].y), img_size.y, 0.5f * img_size.y)
    };

    // Retrieve the complex L elements
    cuFloatComplex l11_c = L_elements[idx].x; // scale_x (complex)
    cuFloatComplex l21_c = L_elements[idx].y; // covariance_xy (complex)
    cuFloatComplex l22_c = L_elements[idx].z; // scale_y (complex)

    // Construct the 2x2 complex covariance matrix from L.
    // We use cuCmulf and cuCaddf for complex multiplication and addition.
    cuFloatComplex cov11_c = cuCmulf(l11_c, l11_c);
    cuFloatComplex cov12_c = cuCmulf(l11_c, l21_c);
    cuFloatComplex cov22_c = cuCaddf(cuCmulf(l21_c, l21_c), cuCmulf(l22_c, l22_c));

    // For computing 2D bounds and conics on a real image plane,
    // we typically use the real parts of the complex covariance elements.
    float3 cov2d_real = make_float3(cuCrealf(cov11_c), cuCrealf(cov12_c), cuCrealf(cov22_c));

    float3 conic;
    float radius;
    // Call the (adapted) function to compute conic and radius from real covariance.
    bool ok = compute_cov2d_bounds(cov2d_real, conic, radius);
    if (!ok) {
        // printf("%d: compute_cov2d_bounds failed (e.g., zero determinant)\n", idx);
        return; // zero determinant or other error
    }

    conics[idx] = conic;
    xys[idx] = center; // Store the real-valued projected center
    radii[idx] = (int)radius;

    uint2 tile_min, tile_max;
    get_tile_bbox(center, radius, tile_bounds, tile_min, tile_max);

    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) {
        // printf("%d point bbox outside of bounds or zero area\n", idx);
        return;
    }
    num_tiles_hit[idx] = tile_area;

    // Fixed depth as in original code
    depths[idx] = 0.0f;
}
