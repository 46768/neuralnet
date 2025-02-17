#ifndef FFN_MEMPOOL_H
#define FFN_MEMPOOL_H

#include "vector.h"
#include "matrix.h"

#include "ffn_init.h"

typedef struct {
	Matrix* weights;
	Vector* biases;
	ActivationFn* activation_fn;
	ActivationFnD* activation_fn_d;
	CostFn cost_fn;
	CostFnD cost_fn_d;
	size_t layer_cnt;

	void* data;
} FFNParameterPool;

typedef struct {
	Vector* preactivations;
	Vector* activations;
	size_t layer_cnt;

	void* data;
} FFNPropagationPool;

typedef struct {
	Matrix* gradient_w;
	Vector* gradient_b;
	size_t layer_cnt;

	void* data;
} FFNGradientPool;

typedef struct {
	Matrix* weight_trsp;
	Matrix* err_coef;
	Vector* a_deriv;
	size_t layer_cnt;

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
FFNParameterPool* ffn_init_parameter_pool(FFNParams*);
FFNPropagationPool* ffn_init_propagation_pool(FFNParams*);
FFNGradientPool* ffn_init_gradient_pool(FFNParams*);
FFNIntermediatePool* ffn_init_intermediate_pool(FFNParams*);
FFNMempool* ffn_init_pool(FFNParams*);

// Memory Management
void ffn_deallocate_parameter_pool(FFNParameterPool*);
void ffn_deallocate_propagation_pool(FFNPropagationPool*);
void ffn_deallocate_gradient_pool(FFNGradientPool*);
void ffn_deallocate_intermediate_pool(FFNIntermediatePool*);
void ffn_deallocate_pool(FFNMempool*);

#endif
