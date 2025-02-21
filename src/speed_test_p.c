#include <math.h>
#include <string.h>

#include "random.h"
#include "logger.h"
#include "allocator.h"
#include "file_io.h"

#include "python_interface.h"
#include "python_grapher.h"
#include "python_get_mnist.h"

#include "generator.h"

#include "ffn.h"
#include "ffn_io.h"
#include "optimizer.h"

int main() {
	FileData* param_bin = get_file_write("nmist_param.bin");
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

	// Test the network on data from the range
	for (int i = 0; i < test_ubound; i++) {
		ffn_run(model, test_input[i]);
		printr("Testing %d/%d\r", i + 1, test_ubound);
	}
	newline();

	// Network forward propagation
	vec_dump(ffn_run(model, train_input[0]));
	vec_dump(ffn_run(model, train_input[1]));

	ffn_export_model(model, param_bin);
	close_file(param_bin);

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
	ffn_deallocate_model(model);

	return 0;
}
