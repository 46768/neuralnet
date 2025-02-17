#ifdef SIMD_AVX
#include "vector.h"

#include <string.h>

#include "logger.h"
#include "allocator.h"

#include "avx.h"
#include "avxmm.h"

//////////////
// Creation //
/////////////

// Initalize a vector with a float* and assign it to a Vector*
void vec_init(size_t dimension, float* dat, Vector* vec) {
	vec->dimension = dimension;
	vec->data = dat;
	memset(vec->data, 0, dimension*sizeof(float));
}

// Create a vector with all element to 0
Vector* vec_zero(size_t dimension) {
	Vector* vec = (Vector*)allocate(sizeof(Vector));
	float* dat = (float*)avx_allocate(dimension*sizeof(float));
	vec_init(dimension, dat, vec);

	return vec;
}

//////////////////////
// Vector Operation //
/////////////////////

// Apply a function that operate on both vector
void _vec_apply(Vector* vec1, Vector* vec2, Vector* res, void(*fn)(float*, float*, float*)) {
#ifndef NO_BOUND_CHECK
	if (vec1->dimension != vec2->dimension) {
		fatal("Vector 1 and 2 dimension mismatched: %d to %d", vec1->dimension, vec2->dimension);
	}
	if (vec1->dimension != res->dimension) {
		fatal("Vector 1 and result vector dimension mismatched: %d to %d",
				vec1->dimension, res->dimension);
	}
#endif

	for (size_t i = 0; i < vec1->dimension; i+=8) {
		fn((vec1->data)+i, (vec2->data)+i, (res->data)+i);
	}
}

// Element wise addition in place
void vec_add_ip(Vector* vec1, Vector* vec2, Vector* res) {
	_vec_apply(vec1, vec2, res, avx_add);
}

// Element wise multiplication in place
void vec_mul_ip(Vector* vec1, Vector* vec2, Vector* res) {
	_vec_apply(vec1, vec2, res, avx_mul);
}

// Element wise multiplication with scalar coefficient in place
void vec_coef_add_ip(Vector* vec1, Vector* vec2, float coef, Vector* res) {
#ifndef NO_BOUND_CHECK
	if (vec1->dimension != vec2->dimension) {
		fatal("Mismatched vec1:vec2 dimension %zu:%zu", vec1->dimension, vec2->dimension);
	}
	if (vec1->dimension != res->dimension) {
		fatal("Mismatched vec1:res dimension %zu:%zu", vec1->dimension, res->dimension);
	}
#endif

	AVX256 vcoef = avxmm256_load_single_ptr(coef), v1dat, v2dat;

	for (int i = 0; i < (int)(vec1->dimension); i+=8) {
		v1dat = avxmm256_load_ptr(&(vec1->data[i]));
		v2dat = avxmm256_load_ptr(&(vec2->data[i]));
		avxmm256_unload_ptr(avxmm256_madd(v1dat, vcoef, v2dat), &(res->data[i]));
	}
}
#endif
