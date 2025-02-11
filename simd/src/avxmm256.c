#include "avxmm256.h"

// Loading/Unloading

__m256 avx_load_ptr_ps(float* ptr) {
	return _mm256_load_ps(ptr);
}

void avx_unload_ptr_ps(__m256 data256, float* ptr) {
	_mm256_store_ps(ptr, data256);
}

// Operation

__m256 avxmm256_add(__m256 a256, __m256 b256) {
	return _mm256_add_ps(a256, b256);
};
__m256 avxmm256_mul(__m256 a256, __m256 b256) {
	return _mm256_mul_ps(a256, b256);
};
#ifdef SIMD_AVX2
__m256 avxmm256_madd(__m256 a256, __m256 b256, __m256 c256) {
	return _mm256_fmadd_ps(a256, b256, c256);
};
#else
__m256 avxmm256_madd(__m256 a256, __m256 b256, __m256 c256) {
	return _mm256_add_ps(_mm256_mul_ps(a256, b256), c256);
	return _mm256_fmadd_ps(a256, b256, c256);
};
#endif
