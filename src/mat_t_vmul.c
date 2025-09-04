#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "avx.h"
#include "matrix.h"
#include "vector.h"

#define SX 3
#define SY 3

int main() {
    MatrixTranpose *mat = (MatrixTranpose *)malloc(sizeof(MatrixTranpose));
    Vector *vec = (Vector *)malloc(sizeof(Vector));
    Vector *res = (Vector *)malloc(sizeof(Vector));

    mat->sx = SX;
    mat->sy = SY;
    mat->data = avx_allocate(((SX + 7) & ~7) * ((SY + 7) & ~7) * sizeof(float));
    mat->data_t = avx_allocate(((SY + 7) & ~7) * ((SX + 7) & ~7) * sizeof(float));

    vec->size = SX;
    vec->data = avx_allocate(((SX + 7) & ~7) * sizeof(float));

    res->size = SY;
    res->data = avx_allocate(((SY + 7) & ~7) * sizeof(float));

    get_mat_t_ctx(m, mat);
    get_vec_ctx(v, vec);

    int cnt = 1;
    for (int i = 0; i < SY; i++) {
        for (int j = 0; j < SX; j++) {
            mat_idx(m, j, i) = cnt;
            cnt++;
        }
    }
	mat_phy_transpose(mat);
    for (int j = 0; j < SX; j++) {
        vec_idx(v, j) = (float)j + 1;
    }

	mat_t_print(mat);
	vec_print(vec);

    clock_t s, e;
    s = clock();
    mat_t_vmul(mat, vec, res);
    e = clock();
    double t = ((double)(e - s)) / CLOCKS_PER_SEC;

    printf("%f\n", t);

	vec_print(res);

    free(mat->data);
    free(mat->data_t);
    free(vec->data);
    free(res->data);
    free(mat);
    free(vec);
    free(res);

    return 0;
}
