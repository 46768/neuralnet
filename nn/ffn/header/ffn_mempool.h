#ifndef FFN_MEMPOOL_H
#define FFN_MEMPOOL_H

#include "ffn_type.h"

#include "vector.h"
#include "matrix.h"

typedef struct {
	// Node Activation
	size_t layer_cnt;
	Vector** preactivations;
	Vector** activations;

	// Backpropagation
	Matrix** gradient_w;
	Vector** gradient_b;
	Vector* gradient_aL_C;
	Vector* a_deriv_L;

	// Backpropagation Temp ptr
	Matrix** weight_trsp;
	Vector** a_deriv;
	Matrix** err_coef;
} FFNMempool;

// Creation
FFNMempool* ffn_init_pool(FFN*);

// Memory Management
void ffn_deallocate_pool(FFNMempool*);

#endif
