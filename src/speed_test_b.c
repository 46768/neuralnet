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
	FFNModel* model = ffn_new_model();
	ffn_add_dense(model, 784, Sigmoid, Xavier, RandomEN2);
	ffn_add_dense(model, 8, Sigmoid, Xavier, RandomEN2);
	ffn_add_dense(model, 8, Sigmoid, Xavier, RandomEN2);
	//ffn_add_dense(model, 64, Sigmoid, Xavier, RandomEN2);
	ffn_add_dense(model, 10, None, Zero, Zero);
	ffn_add_passthrough(model, Softmax);
	ffn_add_passthrough(model, None);
	ffn_set_cost_fn(model, CCE);
	ffn_finalize(model);

	float learning_rate = 0.01;
	ffn_train(model, train_input, train_target, train_ubound, learning_rate, -1);

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
