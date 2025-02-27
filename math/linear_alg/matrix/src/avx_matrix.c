#ifdef SIMD_AVX
#include "matrix.h"

#include <string.h>

#include "avx.h"
#include "avxmm.h"

#include "vector.h"

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
	memset(mat->data, 0, padded_sx*padded_sy*sizeof(float));
}

// Create a matrix with all element to 0
Matrix* matrix_zero(size_t sx, size_t sy) {
	Matrix* mat = (Matrix*)allocate(sizeof(Matrix));
	size_t padded_sx = matrix_calc_ssize(sx);
	size_t padded_sy = matrix_calc_ssize(sy);
	float* dat = (float*)avx_allocate(padded_sx*padded_sy*sizeof(float));
	matrix_init(sx, sy, dat, mat);

	return mat;
}

//////////////////////
// Matrix Operation //
/////////////////////

static inline void _tranpose_kernel(Matrix* mat, Matrix* res, int off_x, int off_y) {
#pragma GCC unroll 8
	for (int x = 0; x < 8; x++) {
#pragma GCC unroll 8
		for (int y = 0; y < 8; y++) {
			*matrix_get_ptr(res, off_y+y, off_x+x) = matrix_get(mat, off_x+x, off_y+y);
		}
	}
}

// Transpose a matrix in place
void matrix_transpose_ip(Matrix* mat, Matrix* res) {
#ifndef NO_BOUND_CHECK
	if (res->sx != mat->sy) {
		fatal("Incompatible sy mat: %zu to sx res: %zu", mat->sy, res->sx);
	}
	if (res->sy != mat->sx) {
		fatal("Incompatible sx mat: %zu to sy res: %zu", mat->sx, res->sy);
	}
#endif
	for (int x = 0; x < (int)(mat->sx); x+=8) {
		for (int y = 0; y < (int)(mat->sy); y+=8) {
			_tranpose_kernel(mat, res, x, y);
		}
	}
}

void matrix_coef_add_ip(Matrix* mat1, Matrix* mat2, float coef, Matrix* res) {
#ifndef NO_BOUND_CHECK
	if (mat1->sx != mat2->sx || mat1->sy != mat2->sy) {
		fatal("Mismatched matrix:mat2 size, %zux%zu, %zuy%zu", mat1->sx, mat2->sx, mat2->sy, mat2->sy);
	}
	if (mat1->sx != res->sx || mat1->sy != res->sy) {
		fatal("Mismatched matrix:res size, %zux%zu, %zuy%zu", mat1->sx, res->sx, mat1->sy, res->sy);
	}
#endif

	size_t ddim = mat1->rsx * mat2->rsy;

	AVX256 vcoef = avxmm256_load_single_ptr(coef), m1data, m2data;
	for (int i = 0; i < (int)ddim; i+=8) {
		m1data = avxmm256_load_ptr(&(mat1->data[i]));
		m2data = avxmm256_load_ptr(&(mat2->data[i]));

		avxmm256_unload_ptr(avxmm256_madd(m1data, vcoef, m2data), &(res->data[i]));
	}
}

/////////////////////////////
// Matrix Vector Operation //
////////////////////////////

// Multiply matrix with vector
void matrix_vec_mul_ip(Matrix* mat, Vector* vec, Vector* res) {
#ifndef NO_BOUND_CHECK
	if (mat->sx != vec->dimension) {
		fatal("Expected input vector size: %d, got %d", mat->sx, vec->dimension);
	}
	if (res->dimension != mat->sy) {
		fatal("Expected result vector size: %d, got %d", mat->sy, vec->dimension);
	}
#endif

	for (size_t i = 0; i < mat->sy; i+=8) {
		AVX256 sub_res = avxmm256_load_single_ptr(0);
		AVX256 vec_data256, mat_data256;
		for (size_t j = 0; j < mat->sx; j++) {
			vec_data256 = avxmm256_load_single_ptr(vec->data[j]);
			mat_data256 = avxmm256_load_ptr(matrix_get_ptr(mat, j, i));

			sub_res = avxmm256_madd(mat_data256, vec_data256, sub_res);
		}

		avxmm256_unload_ptr(sub_res, (res->data)+i);
	}
}
void matrix_vec_mul_offset_ip(Matrix* mat, Vector* vec, Vector* offset, Vector* res) {
#ifndef NO_BOUND_CHECK
	if (mat->sx != vec->dimension) {
		fatal("Expected input vector size: %d, got %d", mat->sx, vec->dimension);
	}
	if (res->dimension != mat->sy) {
		fatal("Expected result vector size: %d, got %d", mat->sy, vec->dimension);
	}
#endif

	for (size_t i = 0; i < mat->sy; i+=8) {
		AVX256 sub_res = avxmm256_load_ptr((offset->data)+i);
		AVX256 vec_data256, mat_data256;
		for (size_t j = 0; j < mat->sx; j++) {
			vec_data256 = avxmm256_load_single_ptr(vec->data[j]);
			mat_data256 = avxmm256_load_ptr(matrix_get_ptr(mat, j, i));

			sub_res = avxmm256_madd(mat_data256, vec_data256, sub_res);
		}

		avxmm256_unload_ptr(sub_res, (res->data)+i);
	}
}

// Get hadamard product of vector and matrix in place
void vec_matrix_hadamard_ip(Vector* vec, Matrix* mat, Matrix* res) {
#ifndef NO_BOUND_CHECK
	if (res->sx != mat->sx) {
		fatal("Incompatible sx mat: %zu to sx res: %zu", mat->sx, res->sx);
	}
	if (res->sy != mat->sy) {
		fatal("Incompatible sy mat: %zu to sy res: %zu", mat->sy, res->sy);
	}
#endif

	for (size_t y = 0; y < mat->sy; y+=8) {
		AVX256 vec_coefficient = avxmm256_load_ptr((vec->data)+y);
		for (size_t x = 0; x < mat->sx; x++) {
			avxmm256_unload_ptr(avxmm256_mul(
						avxmm256_load_ptr(matrix_get_ptr(mat, x, y)),
						vec_coefficient
						), matrix_get_ptr(res, x, y));
		}
	}
}

// Multiply column vector with row vector in place
void column_row_vec_mul_ip(Vector* column, Vector* row, Matrix* res) {
#ifndef NO_BOUND_CHECK
	if (res->sx != row->dimension) {
		fatal("Incompatible row dim: %zu to sx res: %zu", row->dimension, res->sx);
	}
	if (res->sy != column->dimension) {
		fatal("Incompatible col dim: %zu to sy res: %zu", column->dimension, res->sy);
	}
#endif

	for (size_t x = 0; x < row->dimension; x++) {
		AVX256 row_cofficient = avxmm256_load_single_ptr(row->data[x]);
		for (size_t y = 0; y < column->dimension; y+=8) {
			avxmm256_unload_ptr(
					avxmm256_mul(row_cofficient, avxmm256_load_ptr((column->data)+y)),
					matrix_get_ptr(res, x, y)
					);
		}
	}
}

#endif
