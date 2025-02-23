#ifdef SIMD_AVX
#ifndef AVXMM_SIMD_H
#define AVXMM_SIMD_H

#include <immintrin.h>

// Vector Type
typedef __m256 AVX256;
typedef __m128 AVX128;

// constants

#define AVX256ZERO _mm256_set1_ps(0.0f)
#define AVX256ONE _mm256_set1_ps(1.0f)

// Loading/Unloading
AVX256 avxmm256_load_ptr(float*);
AVX256 avxmm256_load_single_ptr(float);
void avxmm256_unload_ptr(AVX256, float*);

AVX128 avxmm128_load_ptr(float*);
void avxmm128_unload_ptr(AVX128, float*);

// Arithmetic Operation
AVX256 avxmm256_add(AVX256, AVX256);
AVX256 avxmm256_mul(AVX256, AVX256);
AVX256 avxmm256_madd(AVX256, AVX256, AVX256);

// Repositioning Operation
#define avxmm256_shuffle(a, b, mask) _mm256_shuffle_ps(a, b, mask)
#define avxmm256_blend(a, b, mask) _mm256_blend_ps(a, b, mask)
#define avxmm256_permute2f128(a, b, mask) _mm256_permute2f128_ps(a, b, mask)

// Pack/Unpacking Operation
AVX256 avxmm256_unpacklo(AVX256, AVX256);
AVX256 avxmm256_unpackhi(AVX256, AVX256);

// MinMax Operation

static inline AVX256 avxmm256_max(AVX256 a256, AVX256 b256) {
	return _mm256_max_ps(a256, b256);
}

// Comparison

static inline AVX256 avxmm256_cmp_GT(AVX256 a256, AVX256 b256) {
	return _mm256_cmp_ps(a256, b256, 30);
}

// Bitmasking

static inline AVX256 avxmm256_mask(AVX256 a256, AVX256 b256) {
	return _mm256_and_ps(a256, b256);
}

// Insertion Operation
#define avxmm256_insertmm128(a256, b128, offset) _mm256_insertf128_ps(a256, b128, offset)

// Typecasting Operation
AVX256 avxmm256_castmm128(AVX128);

#endif
#endif
