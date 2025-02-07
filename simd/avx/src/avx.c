#ifdef SIMD_AVX2
#include "avx.h"

#include <stdlib.h>
#include <immintrin.h>

#include "logger.h"

void* avx_allocate(size_t size) {
	void* ptr;
	size_t padded_size = (size + 31) & ~31;
	if (posix_memalign(&ptr, 32, size+padded_size) != 0) {
		fatal("Failed AVX allocation");
	}
	return ptr;
}

void avx_add(float* a, float* b, float* res) {
	if (a == NULL) {
		fatal("a is null");
	}
	if (b == NULL) {
		fatal("b is null");
	}
	if (res == NULL) {
		fatal("res is null");
	}
	__m256 a256 = _mm256_load_ps(a);
	__m256 b256 = _mm256_load_ps(b);
	__m256 res256 = _mm256_add_ps(a256, b256);
	_mm256_store_ps(res, res256);
}
void avx_mul(float* a, float* b, float* res) {
	if (a == NULL) {
		fatal("a is null");
	}
	if (b == NULL) {
		fatal("b is null");
	}
	if (res == NULL) {
		fatal("res is null");
	}
	__m256 a256 = _mm256_load_ps(a);
	__m256 b256 = _mm256_load_ps(b);
	__m256 res256 = _mm256_mul_ps(a256, b256);
	_mm256_store_ps(res, res256);
}
void avx_madd(float* a, float* b, float* c, float* res) {
	if (a == NULL) {
		fatal("a is null");
	}
	if (b == NULL) {
		fatal("b is null");
	}
	if (c == NULL) {
		fatal("c is null");
	}
	if (res == NULL) {
		fatal("res is null");
	}
	__m256 a256 = _mm256_load_ps(a);
	__m256 b256 = _mm256_load_ps(b);
	__m256 c256 = _mm256_load_ps(c);
	__m256 res256 = _mm256_fmadd_ps(a256, b256, c256);
	_mm256_store_ps(res, res256);
}
float avx_dot(float* a, float* b) {
	if (a == NULL) {
		fatal("a is null");
	}
	if (b == NULL) {
		fatal("b is null");
	}
	__m256 a256 = _mm256_load_ps(a);
	__m256 b256 = _mm256_load_ps(b);
	__m256 c1_256 = _mm256_mul_ps(a256, b256);
	__m256 c2_256 = _mm256_hadd_ps(c1_256, c1_256);
	__m256 c3_256 = _mm256_hadd_ps(c2_256, c2_256);
	__m128 c3_high = _mm256_extractf128_ps(c3_256, 1);
	__m128 c3_low = _mm256_castps256_ps128(c3_256);
	__m128 c4 = _mm_add_ps(c3_low, c3_high);
	
	return _mm_cvtss_f32(c4);
}
#endif
