/** \file */
#ifndef BLAS_MATRIX_H
#define BLAS_MATRIX_H

#include <stdint.h>

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

#endif
