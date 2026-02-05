#include <stdlib.h>

#include "file_io.h"
#include "logger.h"

#include "ffn.h"
#include "ffn_utils.h"

#include "datasets.h"

int main() {
    FFNInitData *initd = ffn_init();
    ffn_add_layer(initd, 784, Sigmoid, Xavier, RandomEN2);
    ffn_add_layer(initd, 16, Sigmoid, Xavier, RandomEN2);
    ffn_add_layer(initd, 16, Sigmoid, Xavier, RandomEN2);
    ffn_set_output(initd, 10, None);
    ffn_set_cost_fn(initd, CCEL);

    FFNModel *model = (FFNModel *)malloc(sizeof(FFNModel));

	FileData *train_file = file_get_read("train_dataset.bin");
	Dataset *train_dataset = dataset_file(train_file->file_pointer);

	FileData *test_file = file_get_read("test_dataset.bin");
	Dataset *test_dataset = dataset_file(test_file->file_pointer);

    ffn_build(initd, model);

	uint32_t i = 0;
	ffn_run(model, (train_dataset->input)+i);
	ffn_print(model);
	vec_print(train_dataset->target+i);
	vec_print(model->propagation.output);

	for (int j = 0; j < 30; j++) {
		ffn_train(model, train_dataset, 0.01);

		// Test the model
		float loss = 0.0f;
		for (uint32_t k = 0; k < test_dataset->size; k++) {
			ffn_run(model, test_dataset->input+k);
			float closs = model->parameter.cost(model->propagation.activation+(model->layer_cnt-1), test_dataset->target+k);
			loss += closs;
		}
		loss /= test_dataset->size;
		info("Validation epoch loss: %.10f", loss);
		info("-Epoch %d--------------------------------------", j);
	}

	ffn_run(model, (train_dataset->input)+i);
	ffn_print(model);
	vec_print(train_dataset->target+i);
	vec_print(model->propagation.output);

    ffn_free(model);
    ffn_free_init(initd);

	dataset_free(train_dataset);
	dataset_free(test_dataset);

	file_close(train_file);
	file_close(test_file);

    return 0;
}
