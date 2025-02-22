#ifdef SIMD_AVX
#ifndef AVXMM_SIMD_H
#define AVXMM_SIMD_H

#include <immintrin.h>

// Vector Type
typedef __m256 AVX256;
typedef __m128 AVX128;

// Loading/Unloading

static inline AVX256 avxmm256_load_ptr(float* ptr) {
	return _mm256_load_ps(ptr);
}
static inline AVX256 avxmm256_load_single_ptr(float val) {
	return _mm256_set1_ps(val);
}
static inline void avxmm256_unload_ptr(AVX256 data256, float* ptr) {
	_mm256_store_ps(ptr, data256);
}

static inline AVX128 avxmm128_load_ptr(float* ptr) {
	return _mm_load_ps(ptr);
}
static inline void avxmm128_unload_ptr(AVX128 data128, float* ptr) {
	_mm_store_ps(ptr, data128);
}

// Arithmetic Operation

static inline AVX256 avxmm256_add(AVX256 a256, AVX256 b256) {
	return _mm256_add_ps(a256, b256);
};
static inline AVX256 avxmm256_mul(AVX256 a256, AVX256 b256) {
	return _mm256_mul_ps(a256, b256);
};
static inline AVX256 avxmm256_madd(AVX256 a256, AVX256 b256, AVX256 c256) {
#ifdef SIMD_AVX2
	return _mm256_fmadd_ps(a256, b256, c256);
#else
	return _mm256_add_ps(_mm256_mul_ps(a256, b256), c256);
#endif
};

// Repositioning Operation
#define avxmm256_shuffle(a, b, mask) _mm256_shuffle_ps(a, b, mask)
#define avxmm256_blend(a, b, mask) _mm256_blend_ps(a, b, mask)

// Pack/Unpacking Operation

static inline AVX256 avxmm256_unpacklo(AVX256 a256, AVX256 b256) {
	return _mm256_unpacklo_ps(a256, b256);
}
static inline AVX256 avxmm256_unpackhi(AVX256 a256, AVX256 b256) {
	return _mm256_unpackhi_ps(a256, b256);
}

// Insertion Operation
#define avxmm256_insertmm128(a256, b128, offset) _mm256_insertf128_ps(a256, b128, offset)

// Typecasting Operation

static inline AVX256 avxmm256_castmm128(AVX128 a128) {
	return _mm256_castps128_ps256(a128);
}

#endif
#endif
