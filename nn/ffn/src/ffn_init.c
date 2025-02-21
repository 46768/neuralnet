#include "ffn.h"
#include "ffn_init.h"

#include <string.h>

#include "allocator.h"

#ifdef SIMD_AVX
#include "avx.h"
#define _init_alloc(s) avx_allocate(s)
#else
#define _init_alloc(s) allocate(s)
#endif

// Initialization

FFNModel* ffn_new_model() {
	FFNModel* model = (FFNModel*)callocate(1, sizeof(FFNModel));
	FFNParams* init_data = (FFNParams*)allocate(sizeof(FFNParams));

	init_data->hidden_size = 0;
	init_data->hidden_capacity = 10;
	init_data->hidden_layers = (LayerData**)callocate(init_data->hidden_capacity, sizeof(LayerData*));
	init_data->cost_fn_enum = -1;

	model->init_data = init_data;
	model->optimizer = 0;
	model->immutable = FALSE;
	return model;
}

static inline void _ffn_check_immutability(FFNModel* model) {
#ifndef NO_STATE_CHECK
	if (model->immutable) {
		fatal("FFN Model already finalized");
	}
#endif
}
static inline void _ffn_check_cost_fn(FFNModel* model) {
#ifndef NO_STATE_CHECK
	if ((int)(model->init_data->cost_fn_enum) == -1) {
		fatal("Cost function not set yet");
	}
#endif
}

void _ffn_init_layer(FFNParams* init_data, size_t size, LayerType l_type,
		ActivationFNEnum fn_type,
		IniterEnum w_init_type,
		IniterEnum b_init_type
		) {
	size_t hidden_size = init_data->hidden_size;
	if (hidden_size >= init_data->hidden_capacity) {
		init_data->hidden_capacity += 10;
		init_data->hidden_layers = reallocate(
				init_data->hidden_layers,
				init_data->hidden_capacity * sizeof(LayerData*)
				);
	}

	init_data->hidden_layers[hidden_size] = (LayerData*)allocate(sizeof(LayerData));
	LayerData* l = init_data->hidden_layers[hidden_size];
	l->node_cnt = size;
	l->fn_type = fn_type;
	l->w_init_type = w_init_type;
	l->b_init_type = b_init_type;
	l->l_type = l_type;
	init_data->hidden_size++;
}

void ffn_add_dense(FFNModel* model, size_t size, ActivationFNEnum activation_fn,
		IniterEnum w_init, IniterEnum b_init) {
	_ffn_check_immutability(model);
	_ffn_init_layer(model->init_data, size, Dense, activation_fn, w_init, b_init);
}
void ffn_add_passthrough(FFNModel* model, ActivationFNEnum activation_fn) {
	_ffn_check_immutability(model);
	FFNParams* init_data = model->init_data;
	size_t prev_size = init_data->hidden_layers[init_data->hidden_size-1]->node_cnt;
	_ffn_init_layer(model->init_data, prev_size, PassThrough, activation_fn, Zero, Zero);
}
void ffn_add_layer(FFNModel* model, LayerData* l_data) {
	FFNParams* init_data = model->init_data;
	size_t hidden_size = init_data->hidden_size;
	if (hidden_size >= init_data->hidden_capacity) {
		init_data->hidden_capacity += 10;
		init_data->hidden_layers = reallocate(
				init_data->hidden_layers,
				init_data->hidden_capacity * sizeof(LayerData*)
				);
	}

	init_data->hidden_layers[hidden_size] = (LayerData*)allocate(sizeof(LayerData));
	memcpy(init_data->hidden_layers[hidden_size], l_data, sizeof(LayerData));
	init_data->hidden_size++;
}
void ffn_set_cost_fn(FFNModel* model, CostFnEnum cost_fn) {
	_ffn_check_immutability(model);
	model->init_data->cost_fn_enum = cost_fn;
}
void ffn_set_optimizer(FFNModel* model, Optimizer* optimizer) {
	_ffn_check_immutability(model);
	model->optimizer = optimizer;
}
void ffn_set_batch_type(FFNModel* model, BatchTypeEnum batch_type) {
	model->batch_type = batch_type;
}
void ffn_set_batch_size(FFNModel* model, size_t batch_size) {
	model->batch_size = batch_size;
}
void ffn_finalize(FFNModel* model) {
	_ffn_check_cost_fn(model);
	_ffn_check_immutability(model);

	model->papool = ffn_init_parameter_pool(model->init_data);
	model->prpool = ffn_init_propagation_pool(model->init_data);
	model->gpool = ffn_init_gradient_pool(model->init_data);
	model->ipool = ffn_init_intermediate_pool(model->init_data);
	model->optimizer->buf = (float*)_init_alloc(model->gpool->base.data_size);

	if (model->batch_type == FullBatch) {
		model->batch_size = (size_t)(-1);
	} else if (model->batch_type == Stochastic) {
		model->batch_size = 1;
	}
	model->optimizer->batch_size = model->batch_size;

	FFNParams* init_data = model->init_data;
	FFNParameterPool* papool = model->papool;
	LayerData** l_data = init_data->hidden_layers;

	for (int l = 0; l < ((int)papool->base.layer_cnt-1); l++) {
		LayerData* layer_cur = l_data[l];
		LayerData* layer_nxt = l_data[l+1];
		Matrix* weight = &(papool->weights[l]);
		Vector* bias = &(papool->biases[l]);
		size_t sx = layer_cur->node_cnt;
		size_t sy = layer_nxt->node_cnt;

		matrix_iden(weight);
		papool->activation_fn[l] = resolve_activation_fn(layer_cur->fn_type);
		papool->activation_fn_d[l] = resolve_activation_fn_d(layer_cur->fn_type);

		if (layer_nxt->l_type == Dense) {
			Initer w_initer = resolve_initer(layer_cur->w_init_type);
			Initer b_initer = resolve_initer(layer_cur->b_init_type);
			for (size_t y = 0; y < sy; y++) {
				bias->data[y] = b_initer(sx);
				for (size_t x = 0; x < sx; x++) {
					*matrix_get_ptr(weight, x, y) = w_initer(sx);
				}
			}
		}
	}

	papool->cost_fn = resolve_cost_fn(init_data->cost_fn_enum);
	papool->cost_fn_d = resolve_cost_fn_d(init_data->cost_fn_enum);

	model->immutable = TRUE;
}
