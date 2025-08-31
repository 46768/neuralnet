#include <stdio.h>

#include "datasets.h"

#define LB 0
#define UB 10

int main() {
	Dataset *dataset = dataset_linear(LB, UB, 2, -123);
	uint32_t ds_size = dataset->size;

	Vector *in = dataset->input;
	Vector *out = dataset->target;

	for (uint32_t i = 0; i < ds_size; i++) {
		get_vec_ctx(iv, in+i);
		get_vec_ctx(ov, out+i);

		printf("%d: %f %f\n", i, iv.data[0], ov.data[0]);
	}

	dataset_free(dataset);

	return 0;
}
