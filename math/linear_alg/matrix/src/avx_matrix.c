#ifdef SIMD_AVX
#include "matrix.h"

#include <string.h>

#include "avx.h"

#include "vector.h"

#include "logger.h"
#include "allocator.h"

//////////////
// Creation //
/////////////

// Create a matrix with all element to 0
Matrix* matrix_zero(size_t sx, size_t sy) {
	Matrix* mat = (Matrix*)allocate(sizeof(Matrix));
	size_t padded_sx = (sx+7)&-7;
	size_t padded_sy = (sy+7)&-7;
	mat->sx = sx;
	mat->sy = sy;
	mat->data = (float*)avx_allocate(padded_sx*padded_sy*sizeof(float));
	memset(mat->data, 0, sx*sy*sizeof(float));

	return mat;
}

/////////////////////////////
// Matrix Vector Operation //
////////////////////////////

// Multiply matrix with vector
void matrix_vec_mul_ip(Matrix* mat, Vector* vec, Vector* res) {
	if (mat->sx != vec->dimension) {
		fatal("Expected input vector size: %d, got %d", mat->sx, vec->dimension);
	}
	if (res->dimension != mat->sy) {
		fatal("Expected result vector size: %d, got %d", mat->sy, vec->dimension);
	}

	for (size_t i = 0; i < mat->sy; i++) {
		float dot_sum = 0.0f;
		for (size_t j = 0; j < mat->sx; j++) {
			dot_sum += matrix_get(mat, j, i)*vec->data[j];
		}

		res->data[i] = dot_sum;
	}
}
// Multiply matrix with vector
Vector* matrix_vec_mul(Matrix* mat, Vector* vec) {
	Vector* res_vec = vec_zero(mat->sy);
	matrix_vec_mul_ip(mat, vec, res_vec);

	return res_vec;
}

// Get hadamard product of vector and matrix in place
void vec_matrix_hadamard_ip(Vector* vec, Matrix* mat, Matrix* res) {
	size_t sx = mat->sx;
	if (res->sx != mat->sx) {
		fatal("Incompatible sx mat: %zu to sx res: %zu", mat->sx, res->sx);
	}
	if (res->sy != mat->sy) {
		fatal("Incompatible sy mat: %zu to sy res: %zu", mat->sy, res->sy);
	}

	for (size_t y = 0; y < mat->sy; y++) {
		float vec_coefficient = vec->data[y];
		for (size_t x = 0; x < mat->sx; x++) {
			res->data[y*sx + x] = vec_coefficient * matrix_get(mat, x, y);
		}
	}
}
// Get hadamard product of vector and matrix
Matrix* vec_matrix_hadamard(Vector* vec, Matrix* mat) {
	Matrix* res_mat = matrix_dup(mat);
	vec_matrix_hadamard_ip(vec, mat, res_mat);

	return res_mat;
}

// Multiply column vector with row vector in place
void column_row_vec_mul_ip(Vector* column, Vector* row, Matrix* res) {
	if (res->sx != row->dimension) {
		fatal("Incompatible row dim: %zu to sx res: %zu", row->dimension, res->sx);
	}
	if (res->sy != column->dimension) {
		fatal("Incompatible col dim: %zu to sy res: %zu", column->dimension, res->sy);
	}
	size_t sx = row->dimension;

	for (size_t x = 0; x < sx; x++) {
		float row_cofficient = row->data[x];
		for (size_t y = 0; y < column->dimension; y++) {
			res->data[y*sx + x] = row_cofficient * column->data[y];
		}
	}
}
// Multiply column vector with row vector
Matrix* column_row_vec_mul(Vector* column, Vector* row) {
	Matrix* res_mat = matrix_zero(row->dimension, column->dimension);
	column_row_vec_mul_ip(column, row, res_mat);

	return res_mat;
}

#endif
