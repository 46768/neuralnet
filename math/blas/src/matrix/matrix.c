#include "matrix.h"

void mat_init(uint32_t sx, uint32_t sy, float *dptr, Matrix *mptr) {
	mptr->sx = sx;
	mptr->sy = sy;
	mptr->data = dptr;
}
