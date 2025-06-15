#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "avx.h"
#include "matrix.h"
#include "vector.h"

#define SX 16384
#define SY 16384

int main() {
    Matrix *mat = (Matrix *)malloc(sizeof(Matrix));
    Vector *vec = (Vector *)malloc(sizeof(Vector));
    Vector *res = (Vector *)malloc(sizeof(Vector));

    mat->sx = SX;
    mat->sy = SY;
    mat->data = avx_allocate(((SX + 7) & ~7) * ((SY + 7) & ~7) * sizeof(float));

    vec->size = SX;
    vec->data = avx_allocate(((SX + 7) & ~7) * sizeof(float));

    res->size = SY;
    res->data = avx_allocate(((SY + 7) & ~7) * sizeof(float));

    get_mat_ctx(m, mat);
    get_vec_ctx(v, vec);

    int cnt = 1;
    for (int i = 0; i < SY; i++) {
        for (int j = 0; j < SX; j++) {
            mat_idx(m, j, i) = cnt;
            cnt++;
        }
    }
    for (int j = 0; j < SX; j++) {
        vec_idx(v, j) = (float)j + 1;
    }

    /*
    for (int i = 0; i < SX; i++) {
            printf("mat %d: ", i);
            for (int j = 0; j < SY; j++) {
                    printf("%f ", mat_t_idx(m, j, i));
            }
            printf("\n");
    }
    for (int j = 0; j < SX; j++) {
            printf("vec %d: %f\n", j, vec_idx(v, j));
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
    */

    clock_t s, e;
    s = clock();
    mat_vmul(mat, vec, res);
    e = clock();
    double t = ((double)(e - s)) / CLOCKS_PER_SEC;

    printf("%f\n", t);

    /*
    get_vec_ctx(r, res);
    for (int i = 0; i < ((SY+7)&~7); i++) {
            printf("res %d: %f\n", i, vec_idx(r, i));
    }
    */

    free(mat->data);
    free(vec->data);
    free(res->data);
    free(mat);
    free(vec);
    free(res);

    return 0;
}
