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

#include "ffn_init.h"
#include "ffn_fpropagate.h"
#include "ffn_bpropagate.h"
#include "ffn_mempool.h"
#include "ffn_util.h"

int main() {
	FileData* training_csv = get_file_write("nmist_loss.csv");
	fprintf(training_csv->file_pointer, "2,");
	fprintf(training_csv->file_pointer, "Training loss,");
	fprintf(training_csv->file_pointer, "Validation loss,");
	python_create_venv(PROJECT_PATH "/requirements.txt");
	//python_get_mnist(PROJECT_PATH "/data/mnist");
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
	ffn_init_dense(nn, 784, ReLU, He, RandomEN2);
	ffn_init_dense(nn, 16, ReLU, He, RandomEN2);
	ffn_init_dense(nn, 16, ReLU, He, RandomEN2);
	ffn_init_dense(nn, 10, None, Zero, Zero);
	ffn_init_passthru(nn, Softmax);
	ffn_init_passthru(nn, None);
	ffn_set_cost_fn(nn, CCE);
	ffn_init_params(nn);
	FFNMempool* pool = ffn_init_pool(nn);

	float learning_rate = 0.01;
	for (int t = 0; t < 38; t++) {
		for (int i = 0; i < 10; i++) {
			ffn_bpropagate(nn, pool, train_input[i], train_target[i], learning_rate);
			ffn_dump_pool(pool);
			if (pool->activations[pool->layer_cnt-1]->data[0] == NAN) {
			break;
			}
		}
	}
	/*
	for (int t = 0; t < 30; t++) {
		// Sample random range of training data
		int range_lower = floorf(f_random((float)train_lbound, (float)train_ubound));
		int range_upper = floorf(f_random((float)range_lower, (float)train_ubound));

		// Train the network on data from the range
		float train_loss = 0.0f;
		for (int i = range_lower; i < range_upper; i++) {
			printr("Training %d/%d\r", i - range_lower + 1, range_upper - range_lower);
			train_loss += ffn_bpropagate(nn, pool, train_input[i], train_target[i], learning_rate);
		}
		newline();
		train_loss /= range_upper - range_lower;
		fprintf(training_csv->file_pointer, "%.10f,", train_loss);

		// Sample random range of training data
		range_lower = floorf(f_random((float)test_lbound, (float)test_ubound));
		range_upper = floorf(f_random((float)range_lower, (float)test_ubound));

		// Test the network on data from the range
		float test_loss = 0.0f;
		for (int i = range_lower; i < range_upper; i++) {
			printr("Testing %d/%d\r", i - range_lower + 1, range_upper - range_lower);
			ffn_fpropagate(nn, pool, test_input[i]);
			test_loss += nn->cost_fn(pool->activations[nn->hidden_size-1], test_target[i]);
		}
		newline();
		test_loss /= range_upper - range_lower;
		fprintf(training_csv->file_pointer, "%.10f,", test_loss);
		info("Training epoch loss: %.10f", train_loss);
		info("Validation epoch loss: %.10f", test_loss);
		info("-Epoch %d--------------------------------------", t);
	}
	*/
	ffn_dump_data(nn);

	// Network forward propagation
	ffn_fpropagate(nn, pool, train_input[0]);
	ffn_dump_output(pool);
	ffn_fpropagate(nn, pool, train_input[1]);
	ffn_dump_output(pool);

	for (int i = 0; i < 10; i++) {
		int lbl = 0;
		for (int j = 0; j < 10; j++) {
			if (train_target[i]->data[j] == 1.0f) {
				lbl = j;
				break;
			}
		}
		info("Training image for a %d\n", lbl);
		for (int x = 0; x < 28; x++) {
			for (int y = 0; y < 28; y++) {
				if (train_input[i]->data[x*28 + y] > 0.0f) {
					printr("#");
				} else {
					printr(" ");
				}
			}
			newline();
		}
		newline();
	}

	char* fpath = (char*)allocate(strlen(training_csv->filename)+1);
	memcpy(fpath, training_csv->filename, strlen(training_csv->filename));
	fpath[strlen(training_csv->filename)] = '\0';
	fprintf(training_csv->file_pointer, "\n");
	close_file(training_csv);
	info("training filename: %s", fpath);
	//python_graph(fpath);

	// Post train cleanup
	deallocate(fpath);
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
