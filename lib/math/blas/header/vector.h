/** \file */
#ifndef BLAS_VECTOR_H
#define BLAS_VECTOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/**
 * \struct Vector
 * \brief A n-dimensional vector
 *
 * 96 bit wide data
 */
typedef struct {
    uint32_t size; /**< Dimension of the vector - uint32 (32bit)*/
    float *data;   /**< Data pointer of the vector - float* (64bit)*/
} Vector;

void vec_init(uint32_t, float *, Vector *);

#ifdef SIMD_AVX
#define get_vec_rsize(s) (((s) + 7) & ~7)
#else
#define get_vec_rsize(s) s
#endif
#define calc_vec_size(d) get_vec_rsize(d)

// Vector Indexing

#define get_vec_ctx(ctx_name, vec) Vector ctx_name = {(vec)->size, (vec)->data}
#define vec_idx(ctx, i) (ctx.data)[i]
#define vec_idx_ptr(ctx, i) &((ctx.data)[i])

void vec_cadd(Vector *, Vector *, float);
void vec_mul(Vector *, Vector *);

#ifdef __cplusplus
}
#endif

#endif
