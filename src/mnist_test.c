#include "random.h"
#include "logger.h"
#include "allocator.h"
#include "file_io.h"

#include "python_interface.h"
#include "python_get_mnist.h"

#include "generator.h"

#include "ffn.h"
#include "ffn_io.h"

int main() {
	FileData* param_bin = get_file_read("nmist_param.bin");
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
	FFNModel* model = ffn_import_model(param_bin);

	// Test the network on data from the range
	float test_loss = 0.0f;
	for (int i = 0; i < test_ubound; i++) {
		Vector* res = ffn_run(model, test_input[i]);
		test_loss += model->papool->cost_fn(res, test_target[i]);
		printr("Testing %d/%d\r", i + 1, test_ubound);
	}
	newline();
	test_loss /= test_ubound;
	info("Validation epoch loss: %.10f", test_loss);


	// Post train cleanup
	close_file(param_bin);
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
