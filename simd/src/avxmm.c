#ifdef SIMD_AVX
#include "avxmm.h"

#include <immintrin.h>

// Loading/Unloading

AVX256 avxmm256_load_ptr(float* ptr) {
	return _mm256_load_ps(ptr);
}
AVX256 avxmm256_load_single_ptr(float val) {
	return _mm256_set1_ps(val);
}
void avxmm256_unload_ptr(AVX256 data256, float* ptr) {
	_mm256_store_ps(ptr, data256);
}
AVX128 avxmm128_load_ptr(float* ptr) {
	return _mm_load_ps(ptr);
}
void avxmm128_unload_ptr(AVX128 data128, float* ptr) {
	_mm_store_ps(ptr, data128);
}

// Arithmetic Operation

AVX256 avxmm256_add(AVX256 a256, AVX256 b256) {
	return _mm256_add_ps(a256, b256);
};
AVX256 avxmm256_mul(AVX256 a256, AVX256 b256) {
	return _mm256_mul_ps(a256, b256);
};
AVX256 avxmm256_madd(AVX256 a256, AVX256 b256, AVX256 c256) {
#ifdef SIMD_AVX2
	return _mm256_fmadd_ps(a256, b256, c256);
#else
	return _mm256_add_ps(_mm256_mul_ps(a256, b256), c256);
#endif
};

// Pack/Unpacking Operation

AVX256 avxmm256_unpacklo(AVX256 a256, AVX256 b256) {
	return _mm256_unpacklo_ps(a256, b256);
}
AVX256 avxmm256_unpackhi(AVX256 a256, AVX256 b256) {
	return _mm256_unpackhi_ps(a256, b256);
}

// Typecasting Operation

AVX256 avxmm256_castmm128(AVX128 a128) {
	return _mm256_castps128_ps256(a128);
}
#endif
