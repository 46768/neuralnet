#ifndef MATH_MATRIX_H
#define MATH_MATRIX_H

#include <stdlib.h>

#include "vector.h"

// Matrix type definition
typedef struct {
	size_t sx; // Horizontal size
	size_t sy; // Vertical size
	float* data; // 1D data array, access x, y by [y*sx + x]
} Matrix;

// Creation
Matrix* matrix_zero(size_t, size_t); // Create a matrix with all element to 0
Matrix* matrix_rand(size_t, size_t, float, float); // Create a matrix with random values
Matrix* matrix_dup(Matrix*); // Duplicate a matrix


// Matrix operation
float matrix_get(Matrix*, int, int); // Get a value at x, y
Matrix* matrix_transpose(Matrix*); // Transpose a matrix

// Matrix vector operation
Vector* matrix_vec_mul(Matrix*, Vector*); // Multiply matrix with vector
Matrix* vec_matrix_hadamard(Vector*, Matrix*); // Get hadamard product of vector and matrix
Matrix* column_row_vec_mul(Vector*, Vector*); // Multiply column vector with row vector

// Memory management
void matrix_deallocate(Matrix*); // Deallocate a matrix

#endif
