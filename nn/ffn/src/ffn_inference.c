#include "ffn.h"

#include <string.h>

#include "vector.h"
#include "matrix.h"

#include "datasets.h"

void ffn_run(FFNModel *model, Vector *data_in) {
	uint32_t l_cnt = model->layer_cnt;

	MatrixTranpose* weight = model->parameter.weight;
	Vector* bias = model->parameter.bias;
	ActivationFn* activation_fn = model->parameter.activation;

	Vector* activation = model->propagation.activation;
	Vector* preactivation = model->propagation.preactivation;

	memcpy(preactivation->data, data_in->data, data_in->size * sizeof(float));
	memcpy(activation->data, data_in->data, data_in->size * sizeof(float));

	for (uint32_t l = 0; l < l_cnt-1; l++) {
		memcpy(preactivation[l+1].data, bias[l].data, bias[l].size * sizeof(float));
		mat_fmva((Matrix*)(weight+l), activation+l, preactivation+l+1);
		activation_fn[l](preactivation+l+1, activation+l+1);
	}
}

void ffn_train(FFNModel *model, Dataset *dataset) {

}
