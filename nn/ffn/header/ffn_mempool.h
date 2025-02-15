#ifndef FFN_MEMPOOL_H
#define FFN_MEMPOOL_H

#include "ffn_type.h"

#include "vector.h"
#include "matrix.h"

typedef struct {
	Vector* preactivations;
	Vector* activations;

	void* data;
} FFNPropagationPool;

typedef struct {
	Matrix* gradient_w;
	Vector* gradient_b;

	void* data;
} FFNGradientPool;

typedef struct {
	Matrix* weight_trsp;
	Matrix* err_coef;
	Vector* a_deriv;

	void* data;
} FFNIntermediatePool;

typedef struct {
	size_t layer_cnt;

	// Node Activation
	FFNPropagationPool* propagation;

	// Backpropagation
	FFNGradientPool* gradients;

	// Backpropagation Temp ptr
	FFNIntermediatePool* intermediates;
} FFNMempool;

// Creation
FFNPropagationPool* ffn_init_propagation_pool(FFN*);
FFNGradientPool* ffn_init_gradient_pool(FFN*);
FFNIntermediatePool* ffn_init_intermediate_pool(FFN*);
FFNMempool* ffn_init_pool(FFN*);

// Memory Management
void ffn_deallocate_propagation_pool(FFNPropagationPool*);
void ffn_deallocate_gradient_pool(FFNGradientPool*);
void ffn_deallocate_intermediate_pool(FFNIntermediatePool*);
void ffn_deallocate_pool(FFNMempool*);

#endif
