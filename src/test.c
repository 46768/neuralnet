#include <string.h>

#include "logger.h"
#include "random.h"
#include "allocator.h"
#include "file_io.h"

#include "python_interface.h"
#include "python_grapher.h"

#include "generator.h"

#include "ffn_init.h"
#include "ffn_fpropagate.h"
#include "ffn_bpropagate.h"
#include "ffn_mempool.h"
#include "ffn_util.h"

int main() {
	debug("init");
	info(PROJECT_PATH);
	FileData* training_csv = get_file_write("loss.csv");
	python_create_venv(PROJECT_PATH "/requirements.txt");
	init_random();
	// Inputs
	Vector** vecs;
	Vector** targets;
	int REGS_RANGEL = -10;
	int REGS_RANGE = 10;
	//generate_linear_regs(REGS_RANGEL, REGS_RANGE, -4.0f, 5.0f, &vecs, &targets);
	generate_xor(&REGS_RANGEL, &REGS_RANGE, &vecs, &targets);

	FFN* nn = ffn_init();
	ffn_init_dense(nn, 2, Sigmoid, Xavier, RandomEN2);
	ffn_init_dense(nn, 2, Sigmoid, Xavier, RandomEN2);
	ffn_init_dense(nn, 1, None, Zero, Zero);
	ffn_set_cost_fn(nn, BCE);
	FFNMempool* mempool = ffn_init_pool(nn);
	ffn_init_params(nn);

	info("Pre train");
	ffn_dump_data(nn);

	float learning_rate = 0.01;
	float prev_l = 1e10;
	// Trains however many times
	for (int t = 0; t < 100000; t++) {
		float l = 0;
		for (int i = REGS_RANGEL; i <= REGS_RANGE; i++) {
			Vector* x = vecs[i - REGS_RANGEL];
			Vector* y = targets[i - REGS_RANGEL];
			l += ffn_bpropagate(nn, mempool, x, y, learning_rate);
			//info("Accmulated loss: %f", l);
		}
		l /= (REGS_RANGE - REGS_RANGEL + 1);
		fprintf(training_csv->file_pointer, "%.10f,", l);
		if (t % 1 == 0) {
			//info("Training loss: %f", l);
		}
		prev_l = l;
	}

	info("Post train");
	ffn_dump_data(nn);

	info("Testing");
	for (int i = REGS_RANGEL; i <= REGS_RANGE; i++) {
		Vector* x = vecs[i - REGS_RANGEL];
		ffn_fpropagate(nn, mempool, x);
		info("Vec[%d]:", i);
		for (size_t i = 0; i < mempool->activations[nn->hidden_size-1]->dimension; i++) {
			printr("%f", mempool->activations[nn->hidden_size-1]->data[i]);
			newline();
		}
		newline();
	}

	for (int i = REGS_RANGEL; i <= REGS_RANGE; i++) {
		vec_deallocate(vecs[i - REGS_RANGEL]);
		vec_deallocate(targets[i - REGS_RANGEL]);
	}
	deallocate(vecs);
	deallocate(targets);
	ffn_deallocate(nn);
	ffn_deallocate_pool(mempool);

	char* fpath = (char*)allocate(strlen(training_csv->filename));
	memcpy(fpath, training_csv->filename, strlen(training_csv->filename));
	fprintf(training_csv->file_pointer, "\n");
	close_file(training_csv);
	info("training filename: %s", fpath);
	python_graph(fpath);

	return 0;
}
