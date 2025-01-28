#include "generator.h"

#include "vector.h"
#include "allocator.h"

void generate_linear_regs(int lower, int upper, float m, float y, Vector*** vecs, Vector*** targets) {
	*vecs = (Vector**)callocate(upper - lower, sizeof(Vector*));
	*targets = (Vector**)callocate(upper - lower, sizeof(Vector*));
	for (int i = lower; i < upper; i++) {
		(*vecs)[i - lower] = vec_zero(1); (*vecs)[i - lower]->data[0] = i;
		(*targets)[i - lower] = vec_zero(1); (*targets)[i - lower]->data[0] = (m*i) + y;
	}
}

void generate_xor(int* lower, int* upper, Vector*** vecs, Vector*** targets) {
	*lower = 0;
	*upper = 4;
	*vecs = (Vector**)callocate(*upper - *lower, sizeof(Vector*));
	*targets = (Vector**)callocate(*upper - *lower, sizeof(Vector*));

	(*vecs)[0] = vec_zero(2);(*vecs)[0]->data[0]=-1.0f;(*vecs)[0]->data[1]=-1.0f;
	(*vecs)[1] = vec_zero(2);(*vecs)[1]->data[0]=-1.0f;(*vecs)[1]->data[1]=1.0f;
	(*vecs)[2] = vec_zero(2);(*vecs)[2]->data[0]=1.0f;(*vecs)[2]->data[1]=-1.0f;
	(*vecs)[3] = vec_zero(2);(*vecs)[3]->data[0]=1.0f;(*vecs)[3]->data[1]=1.0f;

	(*targets)[0] = vec_zero(1);(*targets)[0]->data[0] = 0.0f;
	(*targets)[1] = vec_zero(1);(*targets)[1]->data[0] = 1.0f;
	(*targets)[2] = vec_zero(1);(*targets)[2]->data[0] = 1.0f;
	(*targets)[3] = vec_zero(1);(*targets)[3]->data[0] = 0.0f;
}
