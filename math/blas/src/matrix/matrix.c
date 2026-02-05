#include "matrix.h"

#include <stdio.h>

void mat_init(uint32_t sx, uint32_t sy, float *dptr, Matrix *mptr) {
    mptr->sx = sx;
    mptr->sy = sy;
    mptr->data = dptr;
}

void mat_t_init(uint32_t sx, uint32_t sy, float *dptr, float *dtptr,
                MatrixTranpose *mptr) {
    mptr->sx = sx;
    mptr->sy = sy;

    mptr->data = dptr;
    mptr->data_t = dtptr;
}

// Matrix Operation

void mat_phy_transpose(MatrixTranpose *mat) {
    get_mat_t_ctx(m, mat);

    for (uint32_t x = 0; x < m.sx; x += 8) {
        for (uint32_t y = 0; y < m.sy; y += 8) {
#pragma GCC unroll 8
            for (int ox = 0; ox < 8; ox++) {
#pragma GCC unroll 8
                for (int oy = 0; oy < 8; oy++) {
                    mat_dt_idx(m, x + ox, y + oy) = mat_idx(m, x + ox, y + oy);
                }
            }
        }
    }
}

void mat_print(Matrix *mat) {
	get_mat_ctx(m, mat);

	printf("Matrix Row*Column: %d*%d\n", m.sx, m.sy);

	for (uint32_t x = 0; x < m.sx; x++) {
		printf("[");
		for (uint32_t y = 0; y < m.sy; y++) {
			printf("%f ", mat_idx(m, x, y));
		}
		printf("]\n");
	}
}

void mat_t_print(MatrixTranpose *mat) {
	get_mat_t_ctx(m, mat);

	printf("Matrix Row*Column: %d*%d\n", m.sx, m.sy);

	printf("Normal Data:\n");
	for (uint32_t y = 0; y < m.sy; y++) {
		printf("[");
		for (uint32_t x = 0; x < m.sx; x++) {
			printf("%f ", mat_idx(m, x, y));
		}
		printf("]\n");
	}

	printf("Transposed Data:\n");
	for (uint32_t x = 0; x < m.sx; x++) {
		printf("[");
		for (uint32_t y = 0; y < m.sy; y++) {
			printf("%f ", mat_dt_idx(m, x, y));
		}
		printf("]\n");
	}
}
