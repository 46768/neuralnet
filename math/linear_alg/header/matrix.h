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
Matrix* matrix_iden(size_t); // Create an identity matrix
Matrix* matrix_rand(size_t, size_t, float, float); // Create a matrix with random values
Matrix* matrix_dup(Matrix*); // Duplicate a matrix


// Matrix operation
float matrix_get(Matrix*, size_t, size_t); // Get a value at x, y

void matrix_transpose_ip(Matrix*, Matrix*); // Transpose a matrix in place
Matrix* matrix_transpose(Matrix*); // Transpose a matrix

// Matrix vector operation
void matrix_vec_mul_ip(Matrix*, Vector*, Vector*); // Multiply matrix with vector in place
Vector* matrix_vec_mul(Matrix*, Vector*); // Multiply matrix with vector
void vec_matrix_hadamard_ip(Vector*, Matrix*, Matrix*); // Get hadamard product of vector and matrix
Matrix* vec_matrix_hadamard(Vector*, Matrix*); // Get hadamard product of vector and matrix
void column_row_vec_mul_ip(Vector*, Vector*, Matrix*); // Multiply column vector with row vector
Matrix* column_row_vec_mul(Vector*, Vector*); // Multiply column vector with row vector

// Memory management
void matrix_deallocate(Matrix*); // Deallocate a matrix

#endif
