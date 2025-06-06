/** \file */
#ifndef BLAS_MATRIX_H
#define BLAS_MATRIX_H

#include <stdint.h>

#include "vector.h"

/**
 * \struct Matrix
 * \brief A nxm-dimensional matrix
 *
 * 128 bit wide data
 */
typedef struct {
	uint32_t sx; /**< Width of the matrix - uint32 (32bit)*/
	uint32_t sy; /**< Height of the matrix - uint32 (32bit)*/
	float* data; /**< Data pointer of the matrix - float* (64bit)*/
} Matrix;

void mat_init(uint32_t, uint32_t, float*, Matrix*);

// Matrix indexing

#define get_mat_ctx(ctx_name, mat) Matrix ctx_name = {mat->sx, mat->sy, mat->data}
#define mat_idx(ctx, x, y) (ctx.data)[x+(y*(ctx.sx))]
#define mat_t_idx(ctx, x, y) (ctx.data)[y+(x*(ctx.sy))]

// Matrix Operation

void mat_vmul(Matrix*, Vector*);
void mat_fmva(Matrix*, Matrix*, Vector*);
void mat_hadamard(Matrix*, Vector*);
void mat_t_hadamard(Matrix*, Vector*);

// Vector Operation (Matrix return)

void vec_crmul(Vector*, Vector*, Matrix*);

#endif
