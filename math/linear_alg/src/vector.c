#include "vector.h"

#include <string.h>

#include "logger.h"
#include "allocator.h"
#include "random.h"

//////////////
// Creation //
/////////////

// Create a vector with all element to 0
Vector* vec_zero(size_t dimension) {
	Vector* vec = (Vector*)allocate(sizeof(Vector));
	vec->dimension = dimension;
	vec->data = (float*)callocate(dimension, sizeof(float));

	return vec;
}

// Create a vector with random values
Vector* vec_rand(size_t dimension, float lb, float ub) {
	Vector* vec = vec_zero(dimension);
	for (int i = 0; i < dimension; i++) {
		vec->data[i] = f_random(lb, ub);
	}
	return vec;
}

// Duplicate a vector
Vector* vec_dup(Vector* src_vec) {
	Vector* vec_clone = vec_zero(src_vec->dimension);
	memcpy(vec_clone->data, src_vec->data, src_vec->dimension*sizeof(float));
	return vec_clone;
}

///////////////////////
// Memory Management //
//////////////////////

// Deallocate a vector
void vec_deallocate(Vector* vec) {
	deallocate(vec->data);
	deallocate(vec);
}

//////////////////////
// Vector Operation //
/////////////////////

// Apply a function that operate on both vector
void _vec_apply(Vector* vec1, Vector* vec2, Vector* res, float(*fn)(float, float)) {
	if (vec1->dimension != vec2->dimension) {
		fatal("Vector 1 and 2 dimension mismatched: %d to %d", vec1->dimension, vec2->dimension);
	}
	if (vec1->dimension != res->dimension) {
		fatal("Vector 1 and result vector dimension mismatched: %d to %d",
				vec1->dimension, res->dimension);
	}
	for (int i = 0; i < vec1->dimension; i++) {
		(res->data)[i] = fn((vec1->data)[i], (vec2->data)[i]);
	}
}

static inline float f_add(float a, float b) { return a+b; }
// Element wise addition in place
void vec_add_ip(Vector* vec1, Vector* vec2, Vector* res) {
	return _vec_apply(vec1, vec2, res, f_add);
}
// Element wise addition
Vector* vec_add(Vector* vec1, Vector* vec2) {
	Vector* res = vec_zero(vec1->dimension);
	vec_add_ip(vec1, vec2, res);
	return res;
}

static inline float f_sub(float a, float b) { return a-b; }
// Element wise subtraction in place
void vec_sub_ip(Vector* vec1, Vector* vec2, Vector* res) {
	return _vec_apply(vec1, vec2, res, f_sub);
}
// Element wise subtraction
Vector* vec_sub(Vector* vec1, Vector* vec2) {
	Vector* res = vec_zero(vec1->dimension);
	vec_sub_ip(vec1, vec2, res);
	return res;
}

static inline float f_mul(float a, float b) { return a*b; }
// Element wise multiplication in place
void vec_mul_ip(Vector* vec1, Vector* vec2, Vector* res) {
	return _vec_apply(vec1, vec2, res, f_mul);
}
// Element wise multiplication
Vector* vec_mul(Vector* vec1, Vector* vec2) {
	Vector* res = vec_zero(vec1->dimension);
	vec_mul_ip(vec1, vec2, res);
	return res;
}

// Perform dot product between 2 vectors
float vec_dot(Vector* vec1, Vector* vec2) {
	if (vec1->dimension != vec2->dimension) {
		fatal("Vector 1 and 2 dimension mismatched: %d to %d", vec1->dimension, vec2->dimension);
	}
	float dot_prod = 0;

	for (int i = 0; i < vec1->dimension; i++) {
		dot_prod += (vec1->data)[i] * (vec2->data)[i];
	}

	return dot_prod;
}
