#include <stdlib.h>
#include <stdio.h>

#include "ffn.h"
#include "datasets.h"

int main() {
    FFNInitData *initd = ffn_init();
    ffn_add_layer(initd, 1, Sigmoid, RandomEN2, RandomEN2);
    ffn_set_output(initd, 1);
    ffn_set_cost_fn(initd, MSE);

    FFNModel *model = (FFNModel *)malloc(sizeof(FFNModel));

	Dataset *dataset = dataset_linear(0, 10, 1, 2);

    ffn_build(initd, model);

	uint32_t i = 2;
	ffn_run(model, (dataset->input)+i);

	printf("%f %f\n", dataset->input[i].data[0], dataset->target[i].data[0]);
	printf("%f\n", model->propagation.activation[model->layer_cnt-1].data[0]);

	ffn_train(model, dataset, 0.01);

	ffn_run(model, (dataset->input)+i);

	printf("%f %f\n", dataset->input[i].data[0], dataset->target[i].data[0]);
	printf("%f\n", model->propagation.activation[model->layer_cnt-1].data[0]);

    ffn_free(model);
    ffn_free_init(initd);
	dataset_free(dataset);

    return 0;
}
