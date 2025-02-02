#include <math.h>

#include "random.h"

#include "python_interface.h"
#include "python_get_mnist.h"

#include "generator.h"

#include "ffn_init.h"
#include "ffn_fpropagate.h"
#include "ffn_mempool.h"
#include "ffn_util.h"

int main() {
	python_create_venv(PROJECT_PATH "/requirements.txt");
	python_get_mnist(PROJECT_PATH "/data/mnist");
	init_random();

	// Dataset
	Vector** train_input;
	Vector** train_target;
	int train_lbound;
	int train_ubound;

	Vector** test_input;
	Vector** test_target;
	int test_lbound;
	int test_ubound;

	// Get MNIST dataset
	generate_mnist(&train_lbound, &train_ubound, &train_input, &train_target,
			"mnist/train_img.idx",
			"mnist/train_label.idx"
			);
	generate_mnist(&test_lbound, &test_ubound, &test_input, &test_target,
			"mnist/test_img.idx",
			"mnist/test_label.idx"
			);

	// Network Building
	FFN* nn = ffn_init();
	ffn_init_dense(nn, 784, Sigmoid, Xavier, RandomEN2);
	ffn_init_dense(nn, 16, Sigmoid, Xavier, RandomEN2);
	ffn_init_dense(nn, 16, Sigmoid, Xavier, RandomEN2);
	ffn_init_dense(nn, 10, None, Zero, Zero);
	ffn_init_passthru(nn, Softmax);
	ffn_init_passthru(nn, None);
	ffn_set_cost_fn(nn, CCE);
	ffn_init_params(nn);
	FFNMempool* pool = ffn_init_pool(nn);

	/*
	float learning_rate = 0.01
	for (int t = 0; t < 10; t++) {
		// Sample random range of training data
		int range_lower = floorf(f_random((float)train_lbound, (float)train_ubound));
		int range_upper = floorf(f_random((float)range_lower, (float)train_ubound));

		// Train the network on data from the range
	}
	*/

	// Network forward propagation
	ffn_fpropagate(nn, pool, train_input[0]);

	ffn_dump_output(pool);

	return 0;
}
