#ifdef SIMD_NONE
#include "matrix.h"

#include <string.h>

#include "vector.h"

#include "logger.h"
#include "allocator.h"

//////////////
// Creation //
/////////////

// Initalize a matrix with float* and assign it to Matrix*
void matrix_init(size_t sx, size_t sy, float* dat, Matrix* mat) {
	size_t padded_sx = matrix_calc_ssize(sx);
	size_t padded_sy = matrix_calc_ssize(sy);
	mat->sx = sx;
	mat->sy = sy;
	mat->rsx = padded_sx;
	mat->rsy = padded_sy;
	mat->data = dat;
}

// Create a matrix with all element to 0
Matrix* matrix_zero(size_t sx, size_t sy) {
	Matrix* mat = (Matrix*)allocate(sizeof(Matrix));
	size_t padded_sx = matrix_calc_ssize(sx);
	size_t padded_sy = matrix_calc_ssize(sy);
	float* dat = (float*)callocate(padded_sx*padded_sy*sizeof(float));
	matrix_init(sx, sy, dat, mat);

	return mat;
}

//////////////////////
// Matrix Operation //
/////////////////////

// Transpose a matrix in place
void matrix_transpose_ip(Matrix* mat, Matrix* res) {
	if (res->sx != mat->sy) {
		fatal("Incompatible sy mat: %zu to sx res: %zu", mat->sy, res->sx);
	}
	if (res->sy != mat->sx) {
		fatal("Incompatible sx mat: %zu to sy res: %zu", mat->sx, res->sy);
	}
	for (int x = 0; x < (int)(mat->sx); x+=2) {
		for (int y = 0; y < (int)(mat->sy); y+=2) {
			*matrix_get_ptr(res, y, x) = matrix_get(mat, x, y);
			*matrix_get_ptr(res, y+1, x) = matrix_get(mat, x, y+1);
			*matrix_get_ptr(res, y, x+1) = matrix_get(mat, x+1, y);
			*matrix_get_ptr(res, y+1, x+1) = matrix_get(mat, x+1, y+1);
		}
	}
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
void matrix_vec_mul_offset_ip(Matrix* mat, Vector* vec, Vector* offset, Vector* res) {
	if (mat->sx != vec->dimension) {
		fatal("Expected input vector size: %d, got %d", mat->sx, vec->dimension);
	}
	if (res->dimension != mat->sy) {
		fatal("Expected result vector size: %d, got %d", mat->sy, vec->dimension);
	}

	for (size_t i = 0; i < mat->sy; i++) {
		float dot_sum = offset->data[i];
		for (size_t j = 0; j < mat->sx; j++) {
			dot_sum += matrix_get(mat, j, i)*vec->data[j];
		}

		res->data[i] = dot_sum;
	}
}

// Get hadamard product of vector and matrix in place
void vec_matrix_hadamard_ip(Vector* vec, Matrix* mat, Matrix* res) {
	if (res->sx != mat->sx) {
		fatal("Incompatible sx mat: %zu to sx res: %zu", mat->sx, res->sx);
	}
	if (res->sy != mat->sy) {
		fatal("Incompatible sy mat: %zu to sy res: %zu", mat->sy, res->sy);
	}

	for (size_t y = 0; y < mat->sy; y++) {
		float vec_coefficient = vec->data[y];
		for (size_t x = 0; x < mat->sx; x++) {
			*matrix_get_ptr(res, x, y) = vec_coefficient * matrix_get(mat, x, y);
		}
	}
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
			*matrix_get_ptr(res, x, y) = row_cofficient * column->data[y];
		}
	}
}
#endif
