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
	deallocate(model);
}

// Running and Training

Vector* ffn_run(FFNModel* model, Vector* input) {
	ffn_fpropagate(model->papool, model->prpool, input);
	return &(model->prpool->activations[model->init_data->hidden_size-1]);
}
float ffn_train(FFNModel* model, Vector** data, Vector** target, size_t d_size, float lr, int max_t) {
	float avg_loss = 0;
	int t_lim = ((max_t == -1) == (max_t < (int)d_size)) ? (int)d_size : max_t;
	for (int t = 0; t < t_lim; t++) {
		Vector* res = ffn_run(model, data[t]);
		avg_loss += model->papool->cost_fn(res, target[t]);
		ffn_get_param_change(model->init_data, model->papool, model->prpool, model->gpool, model->ipool, target[t]);

		ffn_apply_gradient(model->init_data, model->papool, model->gpool, lr);
		printr("Training %d/%d\r", t + 1, t_lim);

	}
	return avg_loss / (max_t == -1 ? (float)d_size : (float)max_t);
}
