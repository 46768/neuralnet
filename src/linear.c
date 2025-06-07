#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include "vector.h"
#include "avx.h"

#define SX 12
#define SY 6

int main() {
	Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
	Vector* vec = (Vector*)malloc(sizeof(Vector));
	Vector* res = (Vector*)malloc(sizeof(Vector));

	mat->sx = SX;
	mat->sy = SY;
	mat->data = avx_allocate(((SX+7)&~7)*((SY+7)&~7)*sizeof(float));

	vec->size = SY;
	vec->data = avx_allocate(((SY+7)&~7)*sizeof(float));

	res->size = SY;
	res->data = avx_allocate(((SY+7)&~7)*sizeof(float));

	get_mat_ctx(m, mat);
	get_vec_ctx(v, vec);

	for (int i = 0; i < SY; i++) {
		vec_idx(v, i) = (float)i+1;
		for (int j = 0; j < SX; j++) {
			mat_idx(m, j, i) = (i*SY)+j+1;
		}
	}

	for (int i = 0; i < SY; i++) {
		printf("vec %d: %f\n", i, vec_idx(v, i));
		printf("mat %d: ", i);
		for (int j = 0; j < SX; j++) {
			printf("%f ", mat_idx(m, j, i));
		}
		printf("\n");
	}

	for (int i = 0; i < ((SX+7)&~7)*((SY+7)&~7); i++) {
		if (i % ((SX+7)&~7) == 0) {
			printf("\n");
		}
		printf("%f ", m.data[i]);
	}
	printf("\n");

	mat_vmul(mat, vec, res);

	get_vec_ctx(r, res);
	for (int i = 0; i < SY; i++) {
		printf("res %d: %f\n", i, vec_idx(r, i));
	}

	free(mat->data);
	free(vec->data);
	free(res->data);
	free(mat);
	free(vec);
	free(res);

	return 0;
}
