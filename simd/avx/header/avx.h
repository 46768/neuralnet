#ifdef SIMD_AVX2
#ifndef AVX_SIMD_H
#define AVX_SIMD_H

#include "stdlib.h"

// Allocation

void* avx_allocate(size_t);

// Operation

void avx_add(float*, float*, float*);
void avx_mul(float*, float*, float*);
void avx_madd(float*, float*, float*, float*);
float avx_dot(float*, float*);

#endif
#endif
