#include "ffn_utils.h"

#include <stdio.h>

#include "vector.h"

void ffn_print(FFNModel *model) {
	printf("{\nModel Data:\n");
	printf("\tLayer count: %d\n", model->layer_cnt);

	printf("\tLayer data:\n");
	for (uint32_t l = 0; l < model->layer_cnt; l++) {
		printf("\t\tLayer %d:\n", l);
		printf("\t\t\tActivation:\n");
		vec_print(model->propagation.activation + l);

		printf("\t\t\tPreactivation:\n");
		vec_print(model->propagation.preactivation + l);

		if (l < model->layer_cnt-1) {
			printf("\t\t\tWeight:\n");
			mat_t_print(model->parameter.weight + l);
		} else {
			printf("\t\t\tWeight: null\n");
		}

		if ((int32_t)(l)-1 >= 0) {
			printf("\t\t\tBias:\n");
			vec_print(model->parameter.bias+l-1);
		} else {
			printf("\t\t\tBias: null\n");
		}
	}
	printf("}\n");

}

void ffn_init_print(FFNInitData *init) {
	printf("{\nInit Data:\n");
	printf("\tLayer count: %d\n", init->layer_cnt);
	printf("}\n");
}

void ffn_print_output(FFNModel *model) {
	vec_print((model->propagation.activation) + (model->layer_cnt-1));
}
