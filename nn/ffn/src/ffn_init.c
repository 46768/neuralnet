#include "ffn_init.h"

#include "activation.h"
#include "cost.h"
#include "initer.h"

#include <stdlib.h>

#include "booltype.h"
#include "logger.h"
#include "allocator.h"


//////////////
// Creation //
//////////////

// Create a feed forward network
FFN* ffn_init() {
	FFN* ffn = (FFN*)allocate(sizeof(FFN));
	ffn->hidden_size = 0;
	ffn->hidden_capacity = 10;
	ffn->hidden_layers = (LayerData**)callocate(ffn->hidden_capacity, sizeof(LayerData*));
	ffn->immutable = FALSE;

	return ffn;
}

//////////////////////
// Network settings //
//////////////////////

// Set network's cost function
void ffn_init_set_cost_fn(FFN* nn, CostFnEnum cost_type) {
#ifndef NO_STATE_CHECK
	if (nn->immutable) {
		error("Unable to modify ffn: Immutable");
		return;
	}
#endif
	nn->cost_fn = resolve_cost_fn(cost_type);
	nn->cost_fn_d = resolve_cost_fn_d(cost_type);
}

//////////////////////////
// Layer Initialization //
//////////////////////////

void _ffn_init_layer(FFN* nn, size_t size, ActivationFNEnum fn_type, LayerType l_type,
		IniterEnum w_init_type, IniterEnum b_init_type) {
#ifndef NO_STATE_CHECK
	if (nn->immutable) {
		error("Unable to modify ffn: Immutable");
		return;
	}
#endif
	size_t hidden_size = nn->hidden_size;
	if (hidden_size >= nn->hidden_capacity) {
		nn->hidden_capacity += 10;
		size_t hidden_cap = nn->hidden_capacity;
		nn->hidden_layers = reallocate(nn->hidden_layers, hidden_cap*sizeof(LayerData*));
	}

	(nn->hidden_layers)[hidden_size] = (LayerData*)allocate(sizeof(LayerData));
	LayerData* layer = (nn->hidden_layers)[hidden_size];
	layer->node_cnt = size;
	layer->fn_type = fn_type;
	layer->l_type = l_type;
	layer->w_init_type = w_init_type;
	layer->b_init_type = b_init_type;
	nn->hidden_size++;
}

// Push a dense (fully connected) layer
void ffn_init_dense(FFN* nn, size_t dense_size, ActivationFNEnum fn_type,
		IniterEnum w_init_type, IniterEnum b_init_type) {
	_ffn_init_layer(nn, dense_size, fn_type, Dense, w_init_type, b_init_type);
}

void ffn_init_passthru(FFN* nn, ActivationFNEnum fn_type) {
	size_t prev_size = nn->hidden_layers[nn->hidden_size-1]->node_cnt;
	_ffn_init_layer(nn, prev_size, fn_type, PassThrough, Zero, Zero);
}

// Finalize a network's layer
void ffn_init_params(FFN* nn) {
#ifndef NO_STATE_CHECK
	if (nn->cost_fn == NULL || nn->cost_fn_d == NULL) {
		error("Network's cost function unset");
		return;
	}
	if (nn->immutable) {
		error("Unable to initialize ffn: Already initialized");
		return;
	}
#endif

	// Allocate space for weights, biases, and activation functions and their derivative
	size_t hidden_cnt = nn->hidden_size-1;
	nn->weights = (Matrix**)allocate(hidden_cnt*sizeof(Matrix*));
	nn->biases = (Vector**)allocate(hidden_cnt*sizeof(Vector*));
	nn->layer_activation = (ActivationFn*)allocate(hidden_cnt*sizeof(ActivationFn));
	nn->layer_activation_d = (ActivationFnD*)allocate(hidden_cnt*sizeof(ActivationFnD));

	// Logs layer for debugging
	LayerData** layer_data = nn->hidden_layers;
	for (size_t l = 0; l < nn->hidden_size; l++) {
		LayerData* layer_cur = layer_data[l];
		debug("Layer Data:");
		debug("Node count: %zu", layer_cur->node_cnt);
		if (layer_cur->l_type == Dense) {
			debug("Layer type: Dense");
		} else if (layer_cur->l_type == PassThrough) {
			debug("Layer type: PassThrough");
		}
		debug("Activation function: %s", resolve_activation_fn_str(layer_cur->fn_type));
	}

	// Initialize the data
	for (size_t l = 0; l < nn->hidden_size-1; l++) {
		LayerData* layer_cur = layer_data[l];
		LayerData* layer_nxt = layer_data[l+1];


		size_t sx = layer_cur->node_cnt;
		size_t sy = layer_nxt->node_cnt;

		// Initialize the data structure
		(nn->weights)[l] = matrix_zero(sx, sy);
		matrix_iden((nn->weights)[l]);
		(nn->biases)[l] = vec_zero(sy);
		(nn->layer_activation)[l] = resolve_activation_fn(layer_cur->fn_type);
		(nn->layer_activation_d)[l] = resolve_activation_fn_d(layer_cur->fn_type);

		// Initialize the actual data
		if (layer_nxt->l_type == Dense) {
			Initer w_initer = resolve_initer(layer_cur->w_init_type);
			Initer b_initer = resolve_initer(layer_cur->b_init_type);
			for (size_t y = 0; y < sy; y++) {
				((nn->biases[l])->data)[y] = b_initer(sx);
				for (size_t x = 0; x < sx; x++) {
					*matrix_get_ptr(nn->weights[l], x, y) = w_initer(sx);
				}
			}
		}
	}

	nn->immutable = TRUE;
}

///////////////////////
// Memory Management //
///////////////////////

// Deallocate a network
void ffn_deallocate(FFN* nn) {
	for (size_t i = 0; i < nn->hidden_size-1; i++) {
		matrix_deallocate(nn->weights[i]);
		vec_deallocate(nn->biases[i]);
		deallocate(nn->hidden_layers[i]);
	}
	deallocate(nn->hidden_layers[nn->hidden_size-1]);
	deallocate(nn->weights);
	deallocate(nn->biases);
	deallocate(nn->hidden_layers);
	deallocate(nn->layer_activation);
	deallocate(nn->layer_activation_d);
	deallocate(nn);
}
