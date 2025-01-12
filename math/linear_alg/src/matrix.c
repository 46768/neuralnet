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

// Create a matrix with random values
Matrix* matrix_rand(size_t sx, size_t sy, float lb, float ub) {
	Matrix* mat = matrix_zero(sx, sy);
	for (int i = 0; i < sx*sy; i++) {
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
float matrix_get(Matrix* mat, int x, int y) {
	if (0 > x && x >= mat->sx) {
		fatal("Matrix index x out of bound");
		exit(1);
	}
	if (0 > y && y >= mat->sy) {
		fatal("Matrix index y out of bound");
		exit(1);
	}
	return mat->data[(y*(mat->sx)) + x];
}

// Transpose a matrix
Matrix* matrix_transpose(Matrix* mat) {
	Matrix* mat_t = matrix_dup(mat);
	mat_t->sx = mat->sy;
	mat_t->sy = mat->sx;
	return mat_t;
}

/////////////////////////////
// Matrix Vector Operation //
////////////////////////////

// Multiply matrix with vector
Vector* matrix_vec_mul(Matrix* mat, Vector* vec) {
	if (mat->sx != vec->dimension) {
		fatal("Expected vector size: %d, got %d", mat->sx, vec->dimension);
		exit(1);
	}

	Vector* res_vec = vec_zero(mat->sy);
	float* res_data = res_vec->data;

	for (int i = 0; i < mat->sy; i++) {
		float dot_sum = 0.0f;
		for (int j = 0; j < mat->sx; j++) {
			dot_sum += matrix_get(mat, j, i)*vec->data[j];
		}

		res_data[i] = dot_sum;
	}

	return res_vec;
}

// Get hadamard product of vector and matrix
Matrix* vec_matrix_hadamard(Vector* vec, Matrix* mat) {
	size_t sx = mat->sx;
	Matrix* res_mat = matrix_dup(mat);

	for (int y = 0; y < mat->sy; y++) {
		float vec_coefficient = vec->data[y];
		for (int x = 0; x < mat->sx; x++) {
			res_mat->data[y*sx + x] *= vec_coefficient;
		}
	}

	return res_mat;
}

// Multiply column vector with row vector
Matrix* column_row_vec_mul(Vector* column, Vector* row) {
	Matrix* res_mat = matrix_zero(row->dimension, column->dimension);
	size_t sx = res_mat->sx;

	for (int x = 0; x < sx; x++) {
		float row_cofficient = row->data[x];
		for (int y = 0; y < res_mat->sy; y++) {
			res_mat->data[y*sx + x] = row_cofficient * column->data[y];
		}
	}

	return res_mat;
}
