#include "matrix.h"

#include <string.h>

#include "logger.h"
#include "random.h"
#include "allocator.h"

//////////////
// Creation //
/////////////

// Create a matrix with all element to 0
Matrix* matrix_zero(size_t sx, size_t sy) {
	Matrix* mat = (Matrix*)allocate(sizeof(Matrix));
	mat->sx = sx;
	mat->sy = sy;
	mat->data = (float*)callocate(sx*sy, sizeof(float));

	return mat;
}

Matrix* matrix_iden(size_t size) {
	Matrix* mat = matrix_zero(size, size);
	for (size_t i = 0; i < size; i++) {
		mat->data[i*size + i] = 1;
	}
	return mat;
}

// Create a identity rectangle matrix
Matrix* matrix_iden_xy(size_t sx, size_t sy) {
	Matrix* mat = matrix_zero(sx, sy);
	size_t min_s = sx < sy ? sx : sy;
	for (size_t i = 0; i < min_s; i++) {
		mat->data[i*sx + i] = 1;
	}
	return mat;
}

// Create a matrix with random values
Matrix* matrix_rand(size_t sx, size_t sy, float lb, float ub) {
	Matrix* mat = matrix_zero(sx, sy);
	for (size_t i = 0; i < sx*sy; i++) {
		mat->data[i] = f_random(lb, ub);
	}

	return mat;
}

// Duplicate a matrix
Matrix* matrix_dup(Matrix* mat) {
	size_t sx = mat->sx, sy = mat->sy;
	Matrix* mat_d = matrix_zero(sx, sy);
	memcpy(mat_d->data, mat->data, sx*sy*sizeof(float));
	return mat_d;
}

///////////////////////
// Memory Management //
//////////////////////

// Deallocate a matrix
void matrix_deallocate(Matrix* mat) {
	deallocate(mat->data);
	deallocate(mat);
}

//////////////////////
// Matrix Operation //
/////////////////////

// Get a value at x, y
inline float matrix_get(Matrix* mat, size_t x, size_t y) {
	if (x >= mat->sx) {
		fatal("Matrix index x out of bound");
	}
	if (y >= mat->sy) {
		fatal("Matrix index y out of bound");
	}
	return mat->data[(y*(mat->sx)) + x];
}

// Transpose a matrix in place
void matrix_transpose_ip(Matrix* mat, Matrix* res) {
	if (res->sx != mat->sy) {
		fatal("Incompatible sy mat: %zu to sx res: %zu", mat->sy, res->sx);
	}
	if (res->sy != mat->sx) {
		fatal("Incompatible sx mat: %zu to sy res: %zu", mat->sx, res->sy);
	}
	memcpy(res->data, mat->data, (mat->sx)*(mat->sy)*sizeof(float));
}
// Transpose a matrix
Matrix* matrix_transpose(Matrix* mat) {
	Matrix* mat_t = matrix_zero(mat->sy, mat->sx);
	matrix_transpose_ip(mat, mat_t);
	return mat_t;
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
