#include <stdlib.h>
#include <stdio.h>

#include "ffn.h"
#include "ffn_utils.h"

#include "datasets.h"
#include "matrix.h"

int main() {
    FFNInitData *initd = ffn_init();
    ffn_add_layer(initd, 1, None, RandomEN2, RandomEN2);
    ffn_set_output(initd, 1);
    ffn_set_cost_fn(initd, MSE);

    FFNModel *model = (FFNModel *)malloc(sizeof(FFNModel));
	Dataset *dataset = dataset_linear(0, 10, 1, 2);

    ffn_build(initd, model);

	ffn_init_print(initd);
	ffn_print(model);
	printf("\n");

	uint32_t i = 10;
	ffn_run(model, (dataset->input)+i);

	printf("%f %f\n", dataset->input[i].data[0], dataset->target[i].data[0]);
	//ffn_print(model);
	ffn_print_output(model);
	printf("%f\n", model->parameter.cost(model->propagation.activation+(model->layer_cnt-1), (dataset->target)+i));
	printf("\n");

	for (int j = 0; j < 10000; j++) {
		ffn_train(model, dataset, 0.01);
	}

	ffn_run(model, (dataset->input)+i);

	printf("%f %f\n", dataset->input[i].data[0], dataset->target[i].data[0]);
	//ffn_print(model);
	ffn_print_output(model);
	printf("%f\n", model->parameter.cost(model->propagation.activation+(model->layer_cnt-1), (dataset->target)+i));
	printf("\n");

    ffn_free(model);
    ffn_free_init(initd);
	dataset_free(dataset);

    return 0;
}
