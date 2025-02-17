#ifndef MATH_MATRIX_H
#define MATH_MATRIX_H

#ifdef SIMD_AVX
#define MATRIX_LIB_TYPE "AVX"
#else
#define MATRIX_LIB_TYPE "Scalar"
#endif

#include "logger.h" // Used in inline function

#include "vector.h"

// Matrix type definition
typedef struct {
	size_t sx; // Horizontal size
	size_t sy; // Vertical size
	size_t rsx; // Real horizontal size
	size_t rsy; // Real vertical size
	float* data; // 1D data array, access x, y by [y*sx + x]
} Matrix;

// Creation
void matrix_init(size_t, size_t, float*, Matrix*); // Initalize a matrix with float* and assign it to Matrix*
Matrix* matrix_zero(size_t, size_t); // Create a matrix with all element to 0

void matrix_iden(Matrix*); // Create an identity matrix
void matrix_rand(float, float, Matrix*); // Create a matrix with random values

// Debugging
void matrix_dump(Matrix*);
void matrix_dump_raw(Matrix*);

// Matrix operation
// Get a value at x, y
static inline float matrix_get(Matrix* mat, size_t x, size_t y) {
#ifndef NO_BOUND_CHECK
	if (x >= mat->rsx || y >= mat->rsy) {fatal("Matrix index out of bound %zux%zu %zuy%zu",
			x,mat->rsx,y,mat->rsy);}
#endif
	return mat->data[(x*(mat->rsy)) + y];
}
// Get a value at x, y
static inline float* matrix_get_ptr(Matrix* mat, size_t x, size_t y) {
#ifndef NO_BOUND_CHECK
	if (x >= mat->rsx || y >= mat->rsy) {fatal("Matrix index out of bound %zux%zu %zuy%zu",
			x,mat->rsx,y,mat->rsy);}
#endif
	return (mat->data)+((x*(mat->rsy)) + y);
}
void matrix_transpose_ip(Matrix*, Matrix*); // Transpose a matrix in place
void matrix_coef_add_ip(Matrix*, Matrix*, float, Matrix*); // Add a matrix multiplied with a coefficent to another matrix


// Memory management
static inline size_t matrix_calc_ssize(size_t size) {
#ifdef SIMD_AVX
	return (size+7)&~7;
#else
	return (size+1)&~1;
#endif
}
/**
 * \brief Returns amount of floats actually allocated
 *
 * Scalar: padded to nearest multiple of 2
 * AVX: padded to nearest multiple of 8
 *
 * @return The amount of floats actually allocated depending on instruction set used
 */
static inline size_t matrix_calc_size(size_t sx, size_t sy) {
#ifdef SIMD_AVX
	return ((sx+7)&~7)*((sy+7)&~7);
#else
	return ((sx+1)&~1)*((sy+1)&~1);
#endif
}
void matrix_deallocate(Matrix*); // Deallocate a matrix

// Matrix vector operation
void matrix_vec_mul_ip(Matrix*, Vector*, Vector*); // Multiply matrix with vector in place
void matrix_vec_mul_offset_ip(Matrix*, Vector*, Vector*, Vector*); // Multiply matrix with vector
																   // with offset in place
void vec_matrix_hadamard_ip(Vector*, Matrix*, Matrix*); // Get hadamard product of vector and matrix
void column_row_vec_mul_ip(Vector*, Vector*, Matrix*); // Multiply column vector with row vector

#endif
