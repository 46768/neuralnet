#include "vector.h"

void vec_init(uint32_t size, float *dptr, Vector *vptr) {
    vptr->data = dptr;
    vptr->size = size;
}
