/** \file */
#ifndef BLAS_MATRIX_H
#define BLAS_MATRIX_H

#ifdef __cplusplus
extern "C" {
#endif

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
    float *data; /**< Data pointer of the matrix - float* (64bit)*/
} Matrix;

typedef struct {
    uint32_t sx;   /**< Width of the matrix - uint32 (32bit)*/
    uint32_t sy;   /**< Height of the matrix - uint32 (32bit)*/
    float *data;   /**< Data pointer of the matrix - float* (64bit)*/
    float *data_t; /**< Data pointer of the transposed matrix - float* (64bit)*/
} MatrixTranpose;

/**
 * \struct MatrixCtx
 * \brief A nxm-dimensional matrix context for optimized indexing
 *
 * 192 bit wide data
 */
typedef struct {
    uint32_t sx;   /**< Width of the matrix - uint32 (32bit)*/
    uint32_t sy;   /**< Height of the matrix - uint32 (32bit)*/
    uint32_t rsx;  /**< Allocated width of the matrix - uint32 (32bit)*/
    uint32_t rsy;  /**< Allocated height of the matrix - uint32 (32bit)*/
    float *data;   /**< Data pointer of the matrix - float* (64bit)*/
    float *data_t; /**< Data pointer of the transpose matrix - float* (64bit)*/
} MatrixCtx;

void mat_init(uint32_t, uint32_t, float *, Matrix *);
void mat_t_init(uint32_t, uint32_t, float *, float *, MatrixTranpose *);

#ifdef SIMD_AVX
#define get_mat_rsize(s) (((s) + 7) & ~7)
#else
#define get_mat_rsize(s) s
#endif
#define calc_mat_size(w, h) (get_mat_rsize(w) * get_mat_rsize(h))

// Matrix indexing

#define get_mat_ctx(ctx_name, mat)                                             \
    MatrixCtx ctx_name = {(mat)->sx,                                           \
                          (mat)->sy,                                           \
                          get_mat_rsize((mat)->sx),                            \
                          get_mat_rsize((mat)->sy),                            \
                          (mat)->data,                                         \
                          NULL}
#define get_mat_t_ctx(ctx_name, mat)                                           \
    MatrixCtx ctx_name = {(mat)->sx,                                           \
                          (mat)->sy,                                           \
                          get_mat_rsize((mat)->sx),                            \
                          get_mat_rsize((mat)->sy),                            \
                          (mat)->data,                                         \
                          (mat)->data_t}

#define mat_idx(ctx, x, y)                                                     \
    (ctx.data)[((y) & 7) + ((x) * 8) + (((y) >> 3) * ctx.rsx * 8)]
#define mat_t_idx(ctx, x, y)                                                   \
    (ctx.data)[((x) & 7) + ((y) * 8) + (((x) >> 3) * ctx.rsy * 8)]
#define mat_dt_idx(ctx, x, y)                                                  \
    (ctx.data_t)[((x) & 7) + ((y) * 8) + (((x) >> 3) * ctx.rsy * 8)]
#define mat_dtt_idx(ctx, x, y)                                                 \
    (ctx.data_t)[((y) & 7) + ((x) * 8) + (((y) >> 3) * ctx.rsx * 8)]

#define mat_idx_ptr(ctx, x, y) &mat_idx(ctx, x, y)
#define mat_t_idx_ptr(ctx, x, y) &mat_t_idx(ctx, x, y)
#define mat_dt_idx_ptr(ctx, x, y) &mat_dt_idx(ctx, x, y)
#define mat_dtt_idx_ptr(ctx, x, y) &mat_dtt_idx(ctx, x, y)

// Matrix Operation

void mat_cadd(Matrix *, Matrix *, float);
void mat_vmul(Matrix *, Vector *, Vector *);
void mat_fmva(Matrix *, Vector *, Vector *);
void mat_t_vmul(MatrixTranpose *, Vector *, Vector *);
void mat_phy_transpose(MatrixTranpose *);

// Vector Operation (Matrix return)

void vec_crmul(Vector *, Vector *, Matrix *);

#ifdef __cplusplus
}
#endif

#endif
