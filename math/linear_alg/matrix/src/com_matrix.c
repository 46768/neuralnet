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
	for (size_t x = 0; x < sx; x++) {
		for (size_t y = 0; y < sy; y++) {
			*matrix_get_ptr(mat, x, y) = f_random(lb, ub);
		}
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

///////////////
// Debugging //
///////////////

void matrix_dump(Matrix* mat) {
	for (size_t y = 0; y < mat->sy; y++) {
		for (size_t x = 0; x < mat->sx; x++) {
			printr("%f ", matrix_get(mat, x, y));
		}
		newline();
	}
}

void matrix_dump_raw(Matrix* mat) {
	for (size_t y = 0; y < mat->rsy; y++) {
		for (size_t x = 0; x < mat->rsx; x++) {
			printr("%f ", matrix_get(mat, x, y));
		}
		newline();
	}
}

///////////////////////
// Memory Management //
//////////////////////

// Deallocate a matrix
void matrix_deallocate(Matrix* mat) {
	deallocate(mat->data);
	deallocate(mat);
}
