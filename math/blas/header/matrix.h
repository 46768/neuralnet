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

/**
 * \struct MatrixCtx
 * \brief A nxm-dimensional matrix context for optimized indexing
 *
 * 192 bit wide data
 */
typedef struct {
	uint32_t sx; /**< Width of the matrix - uint32 (32bit)*/
	uint32_t sy; /**< Height of the matrix - uint32 (32bit)*/
	uint32_t rsx; /**< Allocated width of the matrix - uint32 (32bit)*/
	uint32_t rsy; /**< Allocated height of the matrix - uint32 (32bit)*/
	float* data; /**< Data pointer of the matrix - float* (64bit)*/
} MatrixCtx;

void mat_init(uint32_t, uint32_t, float*, Matrix*);

#ifdef SIMD_AVX
#	define get_mat_rsize(s) (s+7)&~7
#else
#	define get_mat_rsize(s) s
#endif

// Matrix indexing

#define get_mat_ctx(ctx_name, mat) MatrixCtx ctx_name = {mat->sx, mat->sy, get_mat_rsize(mat->sx), get_mat_rsize(mat->sy), mat->data}
#define mat_idx(ctx, x, y) (ctx.data)[x+(y*(ctx.rsx))]
#define mat_t_idx(ctx, x, y) (ctx.data)[y+(x*(ctx.rsy))]

// Matrix Operation

void mat_vmul(Matrix*, Vector*);
void mat_fmva(Matrix*, Matrix*, Vector*);
void mat_hadamard(Matrix*, Vector*);
void mat_t_hadamard(Matrix*, Vector*);

// Vector Operation (Matrix return)

void vec_crmul(Vector*, Vector*, Matrix*);

#endif
