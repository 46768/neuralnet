#include "ffn.h"

#include <string.h>
#include <stdio.h>

#include "matrix.h"
#include "vector.h"

#include "datasets.h"

void ffn_run(FFNModel *model, Vector *data_in) {
    uint32_t l_cnt = model->layer_cnt;

    MatrixTranpose *weight = model->parameter.weight;
    Vector *bias = model->parameter.bias;
    ActivationFn *activation_fn = model->parameter.activation;

    Vector *activation = model->propagation.activation;
    Vector *preactivation = model->propagation.preactivation;

    memcpy(preactivation->data, data_in->data, data_in->size * sizeof(float));
    memcpy(activation->data, data_in->data, data_in->size * sizeof(float));

    for (uint32_t l = 0; l < l_cnt - 1; l++) {
        memcpy(preactivation[l + 1].data, bias[l].data,
               bias[l].size * sizeof(float));
        mat_fmva((Matrix *)(weight + l), activation + l, preactivation + l + 1);
        activation_fn[l](preactivation + l + 1, activation + l + 1);
    }
}

void ffn_train(FFNModel *model, Dataset *dataset, float learning_rate) {
	printf("t\n");
    uint32_t dsize = dataset->size;
    uint32_t l_cnt = model->layer_cnt;

    Vector *d_in = dataset->input;
    Vector *d_target = dataset->target;

    MatrixTranpose *weight = model->parameter.weight;
    Vector *bias = model->parameter.bias;

    Vector *activation = model->propagation.activation;
    Vector *preactivation = model->propagation.preactivation;

    Vector *cost_d_v = model->gradient.cost;
    Vector *bias_g = model->gradient.bias;
    Vector *err_L = bias_g + l_cnt - 1;
    Matrix *weight_g = model->gradient.weight;

    Vector *err_coef = model->intermediate.err_coef;

    ActivationFnD *act_d = model->parameter.activation_d;

    CostFnD cost_d = model->parameter.cost_d;
	printf("t\n");

    for (uint32_t i = 0; i < dsize; i++) {
		printf("e\n");
        // Get model preact + act
        ffn_run(model, d_in + i);
		printf("t\n");

        // Compute gradients

        cost_d(activation + l_cnt - 1, d_target + i, cost_d_v);
        act_d[l_cnt - 2](preactivation + l_cnt - 1, err_L);
		printf("t\n");

        vec_mul(err_L, cost_d_v);
		printf("t\n");

        for (int64_t l = l_cnt - 2; l >= 0; l--) {
			printf("d\n");
            Vector *err_l1 = bias_g + l + 1;

            vec_crmul(err_l1, activation + l, weight_g + l);
			printf("t\n");

            // Next error calculation

			printf("l\n");
			printf("%ld\n", (uintptr_t)(model->intermediate.data));
			printf("%ld\n", (uintptr_t)(err_coef));
			printf("%ld\n", (uintptr_t)(err_coef+l));
			printf("%d\n", (err_coef+l)->size);
			printf("%ld\n", (uintptr_t)((err_coef+l)->data));
			printf("%d\n", (err_l1)->size);
			printf("%ld\n", (uintptr_t)((err_l1)->data));
            mat_t_vmul(weight + l, err_l1, err_coef + l);
			printf("l\n");

			printf("t\n");
            act_d[l](preactivation + l, bias_g + l);
			printf("t\n");
            vec_mul(bias_g + l, err_coef + l);
			printf("t\n");
			printf("f\n");
        }
		printf("t\n");

        // Apply graidents

        for (uint32_t l = 0; l < l_cnt - 1; l++) {
            mat_cadd((Matrix *)(weight + l), weight_g + l, -learning_rate);
            vec_cadd(bias + l, bias_g + l + 1, -learning_rate);

            // Update weight transpose

            mat_phy_transpose(weight + l);
        }
		printf("t\n");
    }
	printf("t\n");
}
