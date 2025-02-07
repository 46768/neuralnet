#include "vector.h"

#include <string.h>

#include "allocator.h"
#include "random.h"

//////////////
// Creation //
/////////////

// Create a vector with random values
Vector* vec_rand(size_t dimension, float lb, float ub) {
	Vector* vec = vec_zero(dimension);
	for (size_t i = 0; i < dimension; i++) {
		vec->data[i] = f_random(lb, ub);
	}
	return vec;
}

// Duplicate a vector
Vector* vec_dup(Vector* src_vec) {
	Vector* vec_clone = vec_zero(src_vec->dimension);
	memcpy(vec_clone->data, src_vec->data, src_vec->dimension*sizeof(float));
	return vec_clone;
}

///////////////////////
// Memory Management //
//////////////////////

// Deallocate a vector
void vec_deallocate(Vector* vec) {
	deallocate(vec->data);
	deallocate(vec);
}
