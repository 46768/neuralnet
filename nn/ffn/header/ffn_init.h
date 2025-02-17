#ifndef FFN_TYPE_H
#define FFN_TYPE_H

#include <stdlib.h>

#include "activation.h"
#include "cost.h"
#include "initer.h"

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
	CostFnEnum cost_fn_enum;
} FFNParams;

#endif
