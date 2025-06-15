#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "avx.h"
#include "matrix.h"
#include "vector.h"

#define SX 16384
#define SY 16384

int main() {
    Vector *vec1 = (Vector *)malloc(sizeof(Vector));
    Vector *vec2 = (Vector *)malloc(sizeof(Vector));
    Matrix *mat = (Matrix *)malloc(sizeof(Matrix));

    mat->sx = SY;
    mat->sy = SX;
    mat->data = avx_allocate(((SX + 7) & ~7) * ((SY + 7) & ~7) * sizeof(float));

    vec1->size = SX;
    vec1->data = avx_allocate(((SX + 7) & ~7) * sizeof(float));

    vec2->size = SY;
    vec2->data = avx_allocate(((SY + 7) & ~7) * sizeof(float));

    get_vec_ctx(v1, vec1);
    get_vec_ctx(v2, vec2);

    for (int j = 0; j < SX; j++) {
        vec_idx(v1, j) = (float)j + 1;
    }

    for (int j = 0; j < SY; j++) {
        vec_idx(v2, j) = (float)j + 3;
    }

    /*
    for (int j = 0; j < SX; j++) {
            printf("vec1 %d: %f\n", j, vec_idx(v1, j));
    }
    printf("\n");
    for (int j = 0; j < SY; j++) {
            printf("vec2 %d: %f\n", j, vec_idx(v2, j));
    }
    */

    clock_t s, e;
    s = clock();
    vec_crmul(vec1, vec2, mat);
    e = clock();
    double t = ((double)(e - s)) / CLOCKS_PER_SEC;

    printf("%f\n", t);

    /*
    get_mat_ctx(m, mat);
    for (int i = 0; i < SX; i++) {
            printf("mat %d: ", i);
            for (int j = 0; j < SY; j++) {
                    printf("%f ", mat_idx(m, j, i));
            }
            printf("\n");
    }
    */

    free(mat->data);
    free(vec1->data);
    free(vec2->data);
    free(mat);
    free(vec1);
    free(vec2);

    return 0;
}
