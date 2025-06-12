#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "matrix.h"
#include "avx.h"

#define SX 784
#define SY 16

int main() {
	MatrixTranpose* mat = (MatrixTranpose*)malloc(sizeof(MatrixTranpose));

	mat->sx = SX;
	mat->sy = SY;
	mat->data = avx_allocate(((SX+7)&~7)*((SY+7)&~7)*sizeof(float));
	mat->data_t = avx_allocate(((SY+7)&~7)*((SX+7)&~7)*sizeof(float));

	get_mat_t_ctx(m, mat);

	int cnt = 1;
	for (int i = 0; i < SY; i++) {
		for (int j = 0; j < SX; j++) {
			mat_idx(m, j, i) = cnt;
			cnt++;
		}
	}

	/*
	for (int i = 0; i < SY; i++) {
		for (int j = 0; j < SX; j++) {
			printf("%f ", mat_idx(m, j, i));
		}
		printf("\n");
	}
	printf("\n");
	*/

	clock_t s, e;
	s = clock();
	mat_phy_transpose(mat);
	e = clock();
	double t = ((double)(e-s)) / CLOCKS_PER_SEC;

	/*
	for (int i = 0; i < SY; i++) {
		for (int j = 0; j < SX; j++) {
			printf("%f ", mat_dt_idx(m, j, i));
		}
		printf("\n");
	}

	for (int i = 0; i < ((SX+7)&~7)*((SY+7)&~7); i++) {
		if (i % 8 == 0) {
			printf("\n");
		}
		if (i % 64 == 0) {
			printf("\n");
		}
		printf("%f ", m.data[i]);
	}
	printf("\n");

	for (int i = 0; i < ((SX+7)&~7)*((SY+7)&~7); i++) {
		if (i % 8 == 0) {
			printf("\n");
		}
		if (i % 64 == 0) {
			printf("\n");
		}
		printf("%f ", m.data_t[i]);
	}
	printf("\n");
	*/

	printf("%f\n", t);

	free(mat->data);
	free(mat->data_t);
	free(mat);

	return 0;
}
