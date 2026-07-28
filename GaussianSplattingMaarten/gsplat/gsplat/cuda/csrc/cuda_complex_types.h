// cuda_complex_types.h
#ifndef CUDA_COMPLEX_TYPES_H
#define CUDA_COMPLEX_TYPES_H

#include <cuComplex.h> // Include CUDA's native complex number header

// --- Complex Number Definitions (using cuComplex.h) ---

// Structure for a 2D complex vector using cuFloatComplex
struct cuFloatComplex2 {
    cuFloatComplex x;
    cuFloatComplex y;
};

// Structure for a 3D complex vector using cuFloatComplex
struct cuFloatComplex3 {
    cuFloatComplex x;
    cuFloatComplex y;
    cuFloatComplex z;
};

#endif // CUDA_COMPLEX_TYPES_H