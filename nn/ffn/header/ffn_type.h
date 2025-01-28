#ifndef FFN_TYPE_H
#define FFN_TYPE_H

#include "activation.h"
#include "cost.h"
#include "initer.h"

#include <stdlib.h>

#include "booltype.h"
#include "matrix.h"
#include "vector.h"

typedef enum {
	Dense,
	PassThrough
} LayerType;

typedef struct {
	size_t node_cnt;
	ActivationFNEnum fn_type;
	IniterEnum w_init_type;
	IniterEnum b_init_type;
	LayerType l_type;
} LayerData;

// Feed forwad network type definition
typedef struct {
	size_t hidden_size;	// Number of layers
	size_t hidden_capacity; // Maximum layer count

	LayerData** hidden_layers; // Layer size array
	booltype immutable; // Network's immutability

	Matrix** weights; // Weight array, weight[0] belongs to layer[0] going forward
	Vector** biases; // Bias array, bias[0] belongs to layer[1]
	
	ActivationFn* layer_activation; // Activation Functions
	ActivationFnD* layer_activation_d; // Activation Function Derivatives
	
	CostFn cost_fn;
	CostFnD cost_fn_d;
} FFN;

#endif
