#ifdef SIMD_AVX
#ifndef AVXMM_SIMD_H
#define AVXMM_SIMD_H

#include <immintrin.h>

// Vector Type
typedef __m256 AVX256;
typedef __m128 AVX128;

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

// Pack/Unpacking Operation
AVX256 avxmm256_unpacklo(AVX256, AVX256);
AVX256 avxmm256_unpackhi(AVX256, AVX256);

// Insertion Operation
#define avxmm256_insertmm128(a256, b128, offset) _mm256_insertf128_ps(a256, b128, offset)

// Typecasting Operation
AVX256 avxmm256_castmm128(AVX128);

#endif
#endif
