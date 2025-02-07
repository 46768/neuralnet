#ifdef SIMD_AVX
#include "vector.h"

#include <string.h>

#include "avx.h"

#include "logger.h"
#include "allocator.h"

//////////////
// Creation //
/////////////

// Create a vector with all element to 0
Vector* vec_zero(size_t dimension) {
	Vector* vec = (Vector*)allocate(sizeof(Vector));
	vec->dimension = dimension;
	vec->data = (float*)avx_allocate(dimension*sizeof(float));
	memset(vec->data, 0, dimension*sizeof(float));

	return vec;
}

//////////////////////
// Vector Operation //
/////////////////////

// Apply a function that operate on both vector
void _vec_apply(Vector* vec1, Vector* vec2, Vector* res, void(*fn)(float*, float*, float*)) {
	if (vec1->dimension != vec2->dimension) {
		fatal("Vector 1 and 2 dimension mismatched: %d to %d", vec1->dimension, vec2->dimension);
	}
	if (vec1->dimension != res->dimension) {
		fatal("Vector 1 and result vector dimension mismatched: %d to %d",
				vec1->dimension, res->dimension);
	}
	for (size_t i = 0; i < vec1->dimension; i+=8) {
		fn((vec1->data)+i, (vec2->data)+i, (res->data)+i);
	}
}

// Element wise addition in place
void vec_add_ip(Vector* vec1, Vector* vec2, Vector* res) {
	return _vec_apply(vec1, vec2, res, avx_add);
}
// Element wise addition
Vector* vec_add(Vector* vec1, Vector* vec2) {
	Vector* res = vec_zero(vec1->dimension);
	vec_add_ip(vec1, vec2, res);
	return res;
}

// Element wise multiplication in place
void vec_mul_ip(Vector* vec1, Vector* vec2, Vector* res) {
	return _vec_apply(vec1, vec2, res, avx_mul);
}
// Element wise multiplication
Vector* vec_mul(Vector* vec1, Vector* vec2) {
	Vector* res = vec_zero(vec1->dimension);
	vec_mul_ip(vec1, vec2, res);
	return res;
}
#endif
