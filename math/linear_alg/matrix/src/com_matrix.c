#include "matrix.h"

#include <stdlib.h>
#include <string.h>

#include "random.h"
#include "allocator.h"
#include "logger.h"

//////////////
// Creation //
/////////////

void matrix_iden(Matrix* mat) {
	memset(mat->data, 0, mat->rsx * mat->rsy);
	size_t min_s = mat->sx < mat->sy ? mat->sx : mat->sy;
	for (size_t i = 0; i < min_s; i++) {
		*matrix_get_ptr(mat, i, i) = 1;
	}
}

// Create a matrix with random values
void matrix_rand(float lb, float ub, Matrix* mat) {
	for (size_t x = 0; x < mat->sx; x++) {
		for (size_t y = 0; y < mat->sy; y++) {
			*matrix_get_ptr(mat, x, y) = f_random(lb, ub);
		}
	}
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
