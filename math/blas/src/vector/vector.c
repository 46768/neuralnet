#include "vector.h"

#include <stdio.h>

void vec_init(uint32_t size, float *dptr, Vector *vptr) {
    vptr->data = dptr;
    vptr->size = size;
}

void vec_print(Vector * vec) {
	get_vec_ctx(v, vec);

	printf("Vector Size: %d\n", v.size);

	printf("[");
	for (uint32_t i = 0; i < v.size; i++) {
		printf("%f ", vec_idx(v, i));
	}
	printf("]\n");
}
