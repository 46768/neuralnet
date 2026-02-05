#include <stdlib.h>

#include "logger.h"

#include "ffn.h"
#include "ffn_utils.h"

#include "datasets.h"

int main() {
    FFNInitData *initd = ffn_init();
    ffn_add_layer(initd, 2, Sigmoid, Xavier, RandomEN2);
    ffn_add_layer(initd, 2, Sigmoid, Xavier, RandomEN2);
    ffn_set_output(initd, 2, None);
    ffn_set_cost_fn(initd, CCEL);

    FFNModel *model = (FFNModel *)malloc(sizeof(FFNModel));
	Dataset *dataset = dataset_xor();
	Dataset *test_dataset = dataset;

    ffn_build(initd, model);

	uint32_t i = 2;
	ffn_run(model, (dataset->input)+i);
	ffn_print(model);
	vec_print(dataset->target+i);
	vec_print(model->propagation.output);

	for (int j = 0; j < 10000000; j++) {
		ffn_train(model, dataset, 0.01);

		// Test the model
		/*
		float loss = 0.0f;
		for (uint32_t k = 0; k < test_dataset->size; k++) {
			ffn_run(model, test_dataset->input+k);
			float closs = model->parameter.cost(model->propagation.activation+(model->layer_cnt-1), test_dataset->target+k);
			loss += closs;
		}
		loss /= test_dataset->size;
		info("Validation epoch loss: %.10f", loss);
		info("-Epoch %d--------------------------------------", j);
		*/
	}

	ffn_run(model, (dataset->input)+i);
	ffn_print(model);
	vec_print(dataset->target+i);
	vec_print(model->propagation.output);

    ffn_free(model);
    ffn_free_init(initd);
	dataset_free(dataset);

    return 0;
}
