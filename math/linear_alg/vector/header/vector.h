#ifndef MATH_VECTOR_H
#define MATH_VECTOR_H

#ifdef SIMD_AVX
#define VECTOR_LIB_TYPE "AVX"
#else
#define VECTOR_LIB_TYPE "Scalar"
#endif

#include <stdlib.h>

// Vector type definition
typedef struct {
	size_t dimension; // Size of the vector
	float* data; // 1D data array
} Vector;

// Creation
Vector* vec_zero(size_t); // Create a vector with all element to 0
Vector* vec_rand(size_t, float, float); // Create a vector with random values
Vector* vec_dup(Vector*); // Duplicate a vector

// Debugging
void vec_dump(Vector*);

// Memory management
void vec_deallocate(Vector*); // Deallocate a vector

// Operation
void vec_add_ip(Vector*, Vector*, Vector*); // Element wise addition in place
void vec_mul_ip(Vector*, Vector*, Vector*); // Element wise multiplication in place
void vec_dot_ip(Vector*, Vector*, Vector*); // Perform dot product between 2 vectors in place

#endif
