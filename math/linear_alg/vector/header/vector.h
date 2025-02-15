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
void vec_init(size_t, float*, Vector*); // Initalize a vector with a float* and assign it to a Vector*
Vector* vec_zero(size_t); // Create a vector with all element to 0

void vec_rand(float, float, Vector*); // Create a vector with random values

// Debugging
void vec_dump(Vector*);

// Memory management

/**
 * \brief Returns amount of floats actually allocated
 *
 * Scalar: no modification
 * AVX: padded to nearest multiple of 8
 *
 * @return The amount of floats actually allocated depending on instruction set used
 */
static inline size_t vec_calc_size(size_t size) {
#ifdef SIMD_AVX
	return (size+7)&~7;
#else
	return size;
#endif
}
void vec_deallocate(Vector*); // Deallocate a vector

// Operation
void vec_add_ip(Vector*, Vector*, Vector*); // Element wise addition in place
void vec_mul_ip(Vector*, Vector*, Vector*); // Element wise multiplication in place
void vec_dot_ip(Vector*, Vector*, Vector*); // Perform dot product between 2 vectors in place

#endif
