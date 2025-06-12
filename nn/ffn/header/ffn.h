#ifndef NN_FFN_H
#define NN_FFN_H

#include <stdint.h>

#include "matrix.h"
#include "vector.h"

#include "activation.h"
#include "initer.h"
#include "cost.h"

typedef struct {
	uint32_t size;
	ActivationEnum activation_fn;
	IniterEnum w_initier;
	IniterEnum b_initier;
} FFNLayerData;

typedef struct {
	uint32_t layer_cnt;
	uint32_t layer_cap;
	FFNLayerData* layer_data;
	CostEnum cost_fn;
} FFNInitData;

typedef struct {
	Matrix* weights;
	Vector* bias;
	Vector cost;
} FFNGradientBuffer;

typedef struct {
	Vector* preactivation;
	Vector* activation;
} FFNPropagationBuffer;

typedef struct {
	uint32_t layer_cnt;

	ActivationFn* activation;
	ActivationFnD* activation_d;
	CostFn cost;
	CostFnD cost_d;

	MatrixTranpose* weights;
	Vector* bias;

	FFNPropagationBuffer propagations;
	FFNGradientBuffer gradients;
	Vector* layer_deriv;
	Vector* err_coef;

	void* data;
} FFNModel;

FFNInitData* ffn_init();
void ffn_add_layer(FFNInitData*, uint32_t, ActivationEnum, IniterEnum, IniterEnum);
void ffn_set_output(FFNInitData*, uint32_t);
void ffn_set_cost_fn(FFNInitData*, CostEnum);

void ffn_build(FFNInitData*, FFNModel*);
void ffn_free(FFNModel*);

#endif
