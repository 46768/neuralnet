#include "ffn.h"
#include "ffn_utils.h"

#include <string.h>

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
		Vector bias_v = bias[l];

        memcpy(preactivation[l + 1].data, bias_v.data,
               bias_v.size * sizeof(float));
        mat_fmva((Matrix *)(weight + l), activation + l, preactivation + l + 1);
        activation_fn[l](preactivation + l + 1, activation + l + 1);
    }
}

void ffn_train(FFNModel *model, Dataset *dataset, float learning_rate) {
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
    for (uint32_t i = 0; i < dsize; i++) {
        // Get model preact + act
        ffn_run(model, d_in + i);

        // Compute gradients

        cost_d(activation + l_cnt - 1, d_target + i, cost_d_v);
        act_d[l_cnt - 2](preactivation + l_cnt - 1, err_L);

        vec_mul(err_L, cost_d_v);

        for (int64_t l = l_cnt - 2; l >= 0; l--) {
            Vector *err_l1 = bias_g + l + 1;

            vec_crmul(err_l1, activation + l, weight_g + l);

            // Next error calculation

            mat_t_vmul(weight + l, err_l1, err_coef + l);

            act_d[l](preactivation + l, bias_g + l);

            vec_mul(bias_g + l, err_coef + l);
        }

        // Apply graidents

        for (uint32_t l = 0; l < l_cnt - 1; l++) {
            mat_cadd((Matrix *)(weight + l), weight_g + l, -learning_rate);
            vec_cadd(bias + l, bias_g + l + 1, -learning_rate);

            // Update weight transpose

            mat_phy_transpose(weight + l);
        }
    }
}
