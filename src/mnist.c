#include "random.h"
#include "logger.h"
#include "allocator.h"
#include "file_io.h"

#include "python_interface.h"
#include "python_grapher.h"
#include "python_get_mnist.h"

#include "generator.h"
#include "optimizer.h"

#include "ffn.h"
#include "ffn_io.h"
#include "ffn_statistic.h"
#include "ffn_util.h"

int main() {
	FileData* training_csv = get_file_write("nmist_loss.csv");
	FileData* gradient_csv = get_file_write("nmist_grad.csv");
	FileData* param_bin = get_file_write("nmist_param.bin");
	fprintf(training_csv->file_pointer, "2,");
	fprintf(training_csv->file_pointer, "Training loss,");
	fprintf(training_csv->file_pointer, "Validation loss,");

	fprintf(gradient_csv->file_pointer, "6,");
	fprintf(gradient_csv->file_pointer, "784,");
	fprintf(gradient_csv->file_pointer, "16,");
	fprintf(gradient_csv->file_pointer, "16,");
	fprintf(gradient_csv->file_pointer, "10,");
	fprintf(gradient_csv->file_pointer, "10,");
	fprintf(gradient_csv->file_pointer, "10,");

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
	ffn_add_dense(model, 784, ReLU, He, RandomE0);
	ffn_add_dense(model, 64, ReLU, He, RandomE0);
	ffn_add_dense(model, 64, ReLU, He, RandomE0);
	ffn_add_dense(model, 10, None, Zero, Zero);
	ffn_add_passthrough(model, Softmax);
	ffn_add_passthrough(model, None);
	ffn_set_cost_fn(model, CCE);
	ffn_set_batch_type(model, MiniBatch);
	ffn_set_batch_size(model, 100);
	ffn_set_optimizer(model, nn_momentum_optimize_init(0.9));
	//ffn_set_optimizer(model, nn_gradient_descent_init());
	ffn_finalize(model);

	float learning_rate = 0.01;
	for (int t = 0; t < 100; t++) {
		// Train the network on data from the range
		float train_loss = ffn_train(model, train_input, train_target, train_ubound, learning_rate, -1);
		newline();
		//ffn_dump_propagation(model->prpool);
		//ffn_dump_gradient(model->gpool);
		ffn_export_gradient(model, gradient_csv);
		fprintf(training_csv->file_pointer, "%.10f,", train_loss);

		// Test the network on data from the range
		float test_loss = 0.0f;
		for (int i = 0; i < test_ubound; i++) {
			Vector* res = ffn_run(model, test_input[i]);
			test_loss += model->papool->cost_fn(res, test_target[i]);
			printr("Testing %d/%d\r", i + 1, test_ubound);
		}
		newline();
		test_loss /= test_ubound;
		fprintf(training_csv->file_pointer, "%.10f,", test_loss);
		info("Training epoch loss: %.10f", train_loss);
		info("Validation epoch loss: %.10f", test_loss);
		info("-Epoch %d--------------------------------------", t);
	}

	ffn_dump_param(model->papool);

	// Network forward propagation
	vec_dump(ffn_run(model, train_input[0]));
	newline();
	vec_dump(ffn_run(model, train_input[1]));

	fflush(training_csv->file_pointer);
	python_graph(training_csv->filename);
	close_file(training_csv);

	ffn_export_model(model, param_bin);
	close_file(param_bin);

	close_file(gradient_csv);

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
