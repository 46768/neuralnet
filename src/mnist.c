#include <math.h>

#include "random.h"
#include "logger.h"
#include "allocator.h"

#include "python_interface.h"
#include "python_get_mnist.h"

#include "generator.h"

#include "ffn_init.h"
#include "ffn_fpropagate.h"
#include "ffn_bpropagate.h"
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

	ffn_bpropagate(nn, pool, train_input[0], train_target[0], 0.01);
	ffn_bpropagate(nn, pool, train_input[0], train_target[0], 0.01);
	ffn_bpropagate(nn, pool, train_input[0], train_target[0], 0.01);
	/*
	float learning_rate = 0.01;
	for (int t = 0; t < 10; t++) {
		// Sample random range of training data
		int range_lower = floorf(f_random((float)train_lbound, (float)train_ubound));
		int range_upper = floorf(f_random((float)range_lower, (float)train_ubound));

		// Train the network on data from the range
		for (int i = range_lower; i < range_upper; i++) {
			ffn_bpropagate(nn, pool, train_input[i], train_target[i], learning_rate);
		}

		// Sample random range of training data
		range_lower = floorf(f_random((float)test_lbound, (float)test_ubound));
		range_upper = floorf(f_random((float)range_lower, (float)test_ubound));

		// Train the network on data from the range
		float loss = 0.0f;
		for (int i = range_lower; i < range_upper; i++) {
			loss += ffn_bpropagate(nn, pool, test_input[i], test_target[i], learning_rate);
		}
		loss /= range_upper - range_lower + 1;
		info("Training epoch loss: %.10f", loss);
	}
	*/

	// Network forward propagation
	ffn_fpropagate(nn, pool, train_input[0]);
	ffn_dump_output(pool);
	ffn_fpropagate(nn, pool, train_input[1]);
	ffn_dump_output(pool);

	// Post train cleanup
	for (int i = 0; i < train_ubound; i++) {
		vec_deallocate(train_input[i]);
		vec_deallocate(train_target[i]);
	}
	deallocate(train_input);
	deallocate(train_target);

	for (int i = 0; i < test_ubound; i++) {
		vec_deallocate(test_input[i]);
		vec_deallocate(test_target[i]);
	}
	deallocate(test_input);
	deallocate(test_target);

	ffn_deallocate(nn);
	ffn_deallocate_pool(pool);

	return 0;
}
