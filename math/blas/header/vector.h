/** \file */
#ifndef BLAS_VECTOR_H
#define BLAS_VECTOR_H

#include <stdint.h>

/**
 * \struct Vector
 * \brief A n-dimensional vector
 */
typedef struct {
	uint32_t size; /**< Dimension of the vector */
	float* data; /**< Data pointer of the vector */
} Vector;

void vec_init(uint32_t, float*, Vector*);

#endif
