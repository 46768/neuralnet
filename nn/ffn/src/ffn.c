#include "ffn.h"

#include "logger.h"
#include "allocator.h"

#include "ffn_init.h"
#include "ffn_mempool.h"
#include "ffn_fpropagate.h"
#include "ffn_bpropagate.h"

// Creation

FFNModel* ffn_new_model() {
	FFNModel* model = (FFNModel*)callocate(1, sizeof(FFNModel));
	model->nn = ffn_init();
	model->immutable = FALSE;
	return model;
}

// Memory Management

void ffn_deallocate_model(FFNModel* model) {
	ffn_deallocate(model->nn);
	ffn_deallocate_pool(model->pool);
	deallocate(model);
}

// Initialization

void ffn_add_dense(FFNModel* model, size_t size, ActivationFNEnum activation_fn,
		IniterEnum w_init, IniterEnum b_init) {
#ifndef NO_STATE_CHECK
	if (model->immutable) {
		fatal("FFN Model already finalized");
	}
#endif
	ffn_init_dense(model->nn, size, activation_fn, w_init, b_init);
}
void ffn_add_passthrough(FFNModel* model, ActivationFNEnum activation_fn) {
#ifndef NO_STATE_CHECK
	if (model->immutable) {
		fatal("FFN Model already finalized");
	}
#endif
	ffn_init_passthru(model->nn, activation_fn);
}
void ffn_set_cost_fn(FFNModel* model, CostFnEnum cost_fn) {
#ifndef NO_STATE_CHECK
	if (model->immutable) {
		fatal("FFN Model already finalized");
	}
#endif
	ffn_init_set_cost_fn(model->nn, cost_fn);
}
void ffn_set_optimizer(FFNModel* model, OptimizerFnEnum optimizer_fn) {
#ifndef NO_STATE_CHECK
	if (model->immutable) {
		fatal("FFN Model already finalized");
	}
#endif
	model->optimizer = optimizer_fn;
}
void ffn_finalize(FFNModel* model) {
#ifndef NO_STATE_CHECK
	if (model->immutable) {
		fatal("FFN Model already finalized");
	}
#endif

	ffn_init_params(model->nn);
	model->pool = ffn_init_pool(model->nn);
	model->immutable = TRUE;
}

// Running and Training

Vector* ffn_run(FFNModel* model, Vector* input) {
	FFNMempool* pool = model->pool;
	ffn_fpropagate(model->nn, pool->propagation, input);
	return &(pool->propagation->activations[pool->layer_cnt-1]);
}
float ffn_train(FFNModel* model, Vector** data, Vector** target, size_t d_size, float lr, int max_t) {
	float avg_loss = 0;
	int t_lim = ((max_t == -1) == (max_t < (int)d_size)) ? (int)d_size : max_t;
	for (int t = 0; t < t_lim; t++) {
		Vector* res = ffn_run(model, data[t]);
		avg_loss += model->nn->cost_fn(res, target[t]);
		ffn_get_param_change(model->nn, model->pool, target[t]);

		ffn_apply_gradient(model->nn, model->pool->gradients, lr);
		printr("Testing %d/%d\r", t + 1, t_lim);

	}
	return avg_loss / (max_t == -1 ? (float)d_size : (float)max_t);
}
