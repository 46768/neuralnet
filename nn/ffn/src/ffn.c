#include "ffn.h"

#include "logger.h"
#include "allocator.h"

#include "ffn_init.h"
#include "ffn_mempool.h"
#include "ffn_fpropagate.h"
#include "ffn_bpropagate.h"

// Memory Management

void ffn_deallocate_model(FFNModel* model) {
	ffn_deallocate_parameter_pool(model->papool);
	ffn_deallocate_propagation_pool(model->prpool);
	ffn_deallocate_gradient_pool(model->gpool);
	ffn_deallocate_intermediate_pool(model->ipool);
	for (int i = 0; i < (int)(model->init_data->hidden_size); i++) {
		deallocate(model->init_data->hidden_layers[i]);
	}
	deallocate(model->init_data->hidden_layers);
	deallocate(model->init_data);
	deallocate(model->optimizer->config);
	deallocate(model->optimizer->buf);
	deallocate(model->optimizer);
	deallocate(model);
}

// Running and Training

Vector* ffn_run(FFNModel* model, Vector* input) {
	ffn_fpropagate(model->papool, model->prpool, input);
	return &(model->prpool->activations[model->init_data->hidden_size-1]);
}
float ffn_train(FFNModel* model, Vector** data, Vector** target, size_t d_size, float lr, int max_t) {
	Optimizer* optimizer = model->optimizer;
	OptimizerFn optimizer_fn = optimizer->fn;
	OptimizerFn optimizer_finalize = optimizer->finalize;
	void* optimizer_config = optimizer->config;
	float* optimizer_buf = optimizer->buf;

	FFNParams* init_data = model->init_data;
	FFNParameterPool* params = model->papool;
	FFNPropagationPool* propagation = model->prpool;
	FFNGradientPool* gradient = model->gpool;
	FFNIntermediatePool* intermediates = model->ipool;

	float* gradient_dat_ptr = gradient->base.dptr;
	size_t gradient_dat_size = gradient->base.data_size;

	size_t batch_size = model->batch_size;
	if (model->batch_type == FullBatch) {
		optimizer->batch_size = d_size;
	}

	float avg_loss = 0;
	int t_lim = ((max_t == -1) == (max_t < (int)d_size)) ? (int)d_size : max_t;
	for (int t = 0; t < t_lim;) {
		for (size_t tb = 0; tb < batch_size && t < t_lim;) {
			Vector* res = ffn_run(model, data[t]);
			avg_loss += model->papool->cost_fn(res, target[t]);
			ffn_get_param_change(init_data, params, propagation, gradient, intermediates, target[t]);
			optimizer_fn(optimizer_buf, gradient_dat_ptr, gradient_dat_size, optimizer_config);

			tb++;
			t++;
		}

		optimizer_finalize(optimizer_buf, gradient_dat_ptr, gradient_dat_size, optimizer);
		ffn_apply_gradient(init_data, params, gradient, lr);
		printr("Training %d/%d\r", t + 1, t_lim);

	}
	return avg_loss / t_lim;
}
