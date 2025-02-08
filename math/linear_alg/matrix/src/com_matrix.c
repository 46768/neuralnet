#include "matrix.h"

#include <stdlib.h>
#include <string.h>

#include "random.h"
#include "allocator.h"
#include "logger.h"

//////////////
// Creation //
/////////////

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
	return mat->data[mat->major ? (x*(mat->sx)) + y : (y*(mat->sx)) + x];
}

// Get a value at x, y
inline float* matrix_get_ptr(Matrix* mat, size_t x, size_t y) {
	if (x >= mat->sx) {
		fatal("Matrix index x out of bound");
	}
	if (y >= mat->sy) {
		fatal("Matrix index y out of bound");
	}
	return (mat->data)+(mat->major ? (x*(mat->sx)) + y : (y*(mat->sx)) + x);
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
	res->major = !mat->major;
}
// Transpose a matrix
Matrix* matrix_transpose(Matrix* mat) {
	Matrix* mat_t = matrix_zero(mat->sy, mat->sx);
	matrix_transpose_ip(mat, mat_t);
	return mat_t;
}
