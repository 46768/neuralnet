#include "avxmm256.h"

// Loading/Unloading
__m256 avx_load_ptr_ps(float*) {

}

void avx_unload_ptr_ps(__m256, float*) {

}

// Operation
__m256 avxmm256_add(__m256, __m256);
__m256 avxmm256_mul(__m256, __m256);
__m256 avxmm256_madd(__m256, __m256, __m256);
