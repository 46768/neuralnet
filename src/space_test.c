#include "ffn.h"
#include "optimizer.h"

int main() {
	// Network Building
	FFNModel* model = ffn_new_model();
	ffn_add_dense(model, 784, Sigmoid, Xavier, RandomEN2);
	ffn_add_dense(model, 16, Sigmoid, Xavier, RandomEN2);
	ffn_add_dense(model, 16, Sigmoid, Xavier, RandomEN2);
	ffn_add_dense(model, 10, None, Zero, Zero);
	ffn_add_passthrough(model, Softmax);
	ffn_add_passthrough(model, None);
	ffn_set_cost_fn(model, CCE);
	ffn_set_batch_type(model, MiniBatch);
	ffn_set_batch_size(model, 100);
	ffn_set_optimizer(model, nn_momentum_optimize_init(0.99));
	ffn_finalize(model);

	ffn_deallocate_model(model);

	return 0;
}
