#include <string.h>

#include "logger.h"
#include "random.h"
#include "allocator.h"
#include "file_io.h"

#include "python_interface.h"
#include "python_grapher.h"

#include "generator.h"

#include "ffn.h"

int main() {
	debug("init");
	info(PROJECT_PATH);
	FileData* training_csv = get_file_write("loss.csv");
	fprintf(training_csv->file_pointer, "1,");
	fprintf(training_csv->file_pointer, "Training loss,");
	python_create_venv(PROJECT_PATH "/requirements.txt");
	init_random();
	// Inputs
	Vector** vecs;
	Vector** targets;
	int REGS_RANGEL = -10;
	int REGS_RANGE = 10;
	//generate_linear_regs(REGS_RANGEL, REGS_RANGE, -4.0f, 5.0f, &vecs, &targets);
	generate_xor(&REGS_RANGEL, &REGS_RANGE, &vecs, &targets);

	FFNModel* model = ffn_new_model();
	ffn_add_dense(model, 2, Sigmoid, Xavier, RandomEN2);
	ffn_add_dense(model, 2, Sigmoid, Xavier, RandomEN2);
	ffn_add_dense(model, 1, None, Zero, Zero);
	ffn_set_cost_fn(model, BCE);
	ffn_finalize(model);

	for (int t = 0; t < 100000; t++) {
		float loss = ffn_train(model, vecs, targets, 4, 0.01, -1);
		fprintf(training_csv->file_pointer, "%.10f,", loss);
	}

	info("Testing model");
	for (int i = 0; i < 4; i++) {
		vec_dump(ffn_run(model, vecs[i]));
		newline();
	}

	char* fpath = (char*)allocate(strlen(training_csv->filename));
	memcpy(fpath, training_csv->filename, strlen(training_csv->filename));
	fprintf(training_csv->file_pointer, "\n");
	close_file(training_csv);
	info("training filename: %s", fpath);
	python_graph(fpath);

	deallocate(fpath);
	for (int i = 0; i < (REGS_RANGE - REGS_RANGEL); i++) {
		vec_deallocate(vecs[i]);
		vec_deallocate(targets[i]);
	}
	deallocate(vecs);
	deallocate(targets);
	ffn_deallocate_model(model);

	return 0;
}
