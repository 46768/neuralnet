#include "ffn.h"

#include <stdio.h>
#include <stdlib.h>

#include "activation.h"
#include "cost.h"
#include "initer.h"

#ifdef SIMD_AVX
#	include "avx.h"
#	define data_alloc(s) avx_allocate(s)
#	define data_pad(s) ((s+63)&~63)-s
#else
#	define data_alloc(s) malloc(s)
#	define data_pad(s) 0
#endif

FFNInitData* ffn_init() {
	FFNInitData* initd = (FFNInitData*)malloc(sizeof(FFNInitData));
	initd->layer_cnt = 0;
	initd->layer_cap = 1;
	initd->cost_fn = MSE;
	initd->layer_data = (FFNLayerData*)malloc(sizeof(FFNLayerData));

	return initd;
}

void ffn_add_layer(
		FFNInitData* initd,
		uint32_t l_size,
		ActivationEnum activation_fn,
		IniterEnum weight_initer,
		IniterEnum bias_initer) {
	if (initd->layer_cnt >= initd->layer_cap) {
		initd->layer_cap *= 2;
		initd->layer_data = (FFNLayerData*)realloc(initd->layer_data, initd->layer_cap);
	}

	uint32_t layer_cnt = initd->layer_cnt++;
	FFNLayerData layer = initd->layer_data[layer_cnt];
	layer.activation_fn = activation_fn;
	layer.w_initier = weight_initer;
	layer.b_initier = bias_initer;
	layer.size = l_size;
}

void ffn_set_output(FFNInitData* initd, uint32_t o_size) {
	ffn_add_layer(initd, o_size, None, Zero, Zero);
}

void ffn_set_cost_fn(FFNInitData* initd, CostEnum cost_fn) {
	initd->cost_fn = cost_fn;
}

void ffn_build(FFNInitData* initd, FFNModel* model) {
	uint32_t layer_cnt = initd->layer_cnt;
	uint32_t layer_cntn1 = layer_cnt-1;
	model->layer_cnt = layer_cnt;

	model->cost = cost_resolve(initd->cost_fn);
	model->cost_d = cost_d_resolve(initd->cost_fn);

	uint32_t d_off_cnt = (layer_cnt*10)-7;
	uint64_t* d_off = (uint64_t*)malloc(d_off_cnt*sizeof(uint64_t));

	uint32_t bias_mdata_size = layer_cntn1 * sizeof(Vector);
	uint32_t weight_mdata_size = layer_cntn1 * sizeof(MatrixTranpose);

	uint32_t bias_g_mdata_size = layer_cntn1 * sizeof(Vector);
	uint32_t weight_g_mdata_size = layer_cntn1 * sizeof(Matrix);
	uint32_t layer_d_mdata_size = layer_cnt * sizeof(Vector);
	uint32_t err_coef_mdata_size = layer_cntn1 * sizeof(Vector);

	uint32_t preactivation_mdata_size = layer_cnt * sizeof(Vector);
	uint32_t activation_mdata_size = layer_cnt * sizeof(Vector);

	uint32_t activation_fn_data_size = layer_cntn1 * sizeof(ActivationFn);
	uint32_t activation_fn_d_data_size = layer_cntn1 * sizeof(ActivationFnD);

	uint32_t i;
	for (i = 0; i < d_off_cnt; i++) {

	}

	uint64_t mdata_size =
		bias_mdata_size
		+weight_mdata_size
		+bias_g_mdata_size
		+weight_g_mdata_size
		+layer_d_mdata_size
		+err_coef_mdata_size
		+preactivation_mdata_size
		+activation_mdata_size
		+activation_fn_data_size
		+activation_fn_d_data_size;
	uint64_t padding = data_pad(mdata_size);

	printf("%lu\n", mdata_size);
}

void ffn_free(FFNModel* model) {
	free(model->data);
	free(model);
}
