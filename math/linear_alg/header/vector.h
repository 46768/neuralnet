#ifndef MATH_VECTOR_H
#define MATH_VECTOR_H

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

// Operation
Vector* vec_add(Vector*, Vector*); // Element wise addition
Vector* vec_sub(Vector*, Vector*); // Element wise subtraction
Vector* vec_mul(Vector*, Vector*); // Element wise multiplication
float vec_dot(Vector*, Vector*); // Perform dot product between 2 vectors

// Memory management
void vec_deallocate(Vector*); // Deallocate a vector

#endif
