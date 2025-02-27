/** \file */
#ifndef NN_DATASETS_H
#define NN_DATASETS_H

#include <stdint.h>

#include "vector.h"

/**
 * \struct Dataset
 * \brief A struct containing a dataset
 *
 * Contains the dataset metadata and data of input-target
 * pair, input[i] correspond to target[i]
 */
typedef struct {
	uint32_t size; /**< Amount of datapoint pairs in the dataset */
	Vector* input; /**< Input of the dataset */
	Vector* target; /**< Target of the dataset */

	void* data; /**< Data pool of the dataset */
} Dataset;

#endif
