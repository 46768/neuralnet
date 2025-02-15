#include "vector.h"

#include <string.h>

#include "allocator.h"
#include "random.h"
#include "logger.h"

//////////////
// Creation //
/////////////

// Create a vector with random values
void vec_rand(float lb, float ub, Vector* vec) {
	for (size_t i = 0; i < vec->dimension; i++) {
		vec->data[i] = f_random(lb, ub);
	}
}

///////////////
// Debugging //
///////////////

void vec_dump(Vector* vec) {
	for (int i = 0; i < (int)(vec->dimension); i++) {
		printr("%.10f\n", vec->data[i]);
	}
}

///////////////////////
// Memory Management //
//////////////////////

// Deallocate a vector
void vec_deallocate(Vector* vec) {
	deallocate(vec->data);
	deallocate(vec);
}
