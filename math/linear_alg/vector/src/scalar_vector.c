#ifdef SIMD_NONE
#include "vector.h"

#include "logger.h"
#include "allocator.h"

//////////////
// Creation //
/////////////

// Initalize a vector with a float* and assign it to a Vector*
void vec_init(size_t dimension, float* dat, Vector* vec) {
	vec->dimension = dimension;
	vec->data = dat;
}

// Create a vector with all element to 0
Vector* vec_zero(size_t dimension) {
	Vector* vec = (Vector*)allocate(sizeof(Vector));
	float* dat = (float*)callocate(dimension*sizeof(float));
	vec_init(dimension, dat, vec);

	return vec;
}

//////////////////////
// Vector Operation //
/////////////////////

// Apply a function that operate on both vector
void _vec_apply(Vector* vec1, Vector* vec2, Vector* res, float(*fn)(float, float)) {
#ifndef NO_BOUND_CHECK
	if (vec1->dimension != vec2->dimension) {
		fatal("Vector 1 and 2 dimension mismatched: %d to %d", vec1->dimension, vec2->dimension);
	}
	if (vec1->dimension != res->dimension) {
		fatal("Vector 1 and result vector dimension mismatched: %d to %d",
				vec1->dimension, res->dimension);
	}
#endif

	for (size_t i = 0; i < vec1->dimension; i++) {
		(res->data)[i] = fn((vec1->data)[i], (vec2->data)[i]);
	}
}

static inline float f_add(float a, float b) { return a+b; }
// Element wise addition in place
void vec_add_ip(Vector* vec1, Vector* vec2, Vector* res) {
	return _vec_apply(vec1, vec2, res, f_add);
}

static inline float f_mul(float a, float b) { return a*b; }
// Element wise multiplication in place
void vec_mul_ip(Vector* vec1, Vector* vec2, Vector* res) {
	return _vec_apply(vec1, vec2, res, f_mul);
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
	for (int i = 0; i < (int)(vec1->dimension); i++) {
		res->data[i] = (vec1->data[i] * coef) + vec2->data[i];
	}
}
#endif
