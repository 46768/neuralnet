#ifndef FFN_TYPE_H
#define FFN_TYPE_H

#include "activation.h"

#include <stdlib.h>

#include "booltype.h"
#include "matrix.h"
#include "vector.h"

typedef struct {
	size_t node_cnt;
	ActivationFN fn_type;
} LayerData;

// Feed forwad network type definition
typedef struct {
	size_t hidden_size;	// Number of layers
	size_t hidden_capacity; // Maximum layer count

	size_t* hidden_layers; // Layer size array
	booltype immutable; // Network's immutability

	Matrix** weights; // Weight array, weight[0] belongs to layer[0] going forward
	Vector** biases; // Bias array, bias[0] belongs to layer[1]
} FFN;

#endif
