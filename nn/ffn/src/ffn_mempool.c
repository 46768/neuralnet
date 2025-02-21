#include "ffn_mempool.h"

#include "allocator.h"

#include "vector.h"
#include "matrix.h"

#include "activation.h"

#include "ffn_init.h"

#ifdef SIMD_AVX
#include "avx.h"
#define _pool_alloc(s) avx_allocate(s)
#define _pool_pad(s) ((s+63)&~63)-s
#else
#define _pool_alloc(s) callocate(s)
#define _pool_pad(s) 0
#endif

FFNParameterPool* _ffn_init_papool(size_t layer_cnt, size_t mat_size_sum, size_t vec_size_sum, FFNParams* nn) {
	FFNParameterPool* papool = (FFNParameterPool*)allocate(sizeof(FFNParameterPool));
	// ((layer_cnt-1)*sizeof(Matrix)) + ((layer_cnt-1)*sizeof(Vector))
	// + ((layer_cnt-1)*(sizeof(ActivationFn)+sizeof(ActivationFnD)))
	size_t mdata_size = ((layer_cnt-1) * (sizeof(Matrix) + sizeof(Vector)
			+ sizeof(ActivationFn) + sizeof(ActivationFnD)));
	size_t padding = _pool_pad(mdata_size);

	papool->base.data = _pool_alloc(
			mdata_size // Space for matrix + Vector pointer
			+ padding // Padding for alignment
			+ ((mat_size_sum + vec_size_sum) * sizeof(float)) // Space for matrix + vector data
			);

	Matrix* mat_ptr = (Matrix*)(papool->base.data);
	Vector* vec_ptr = (Vector*)(mat_ptr + layer_cnt - 1);
	ActivationFn* act_fn_pptr = (ActivationFn*)(vec_ptr + layer_cnt - 1);
	ActivationFnD* act_fn_d_pptr = (ActivationFnD*)(act_fn_pptr + layer_cnt - 1);

	float* mat_dat_ptr = (float*)((char*)(act_fn_d_pptr + layer_cnt - 1) + padding);
	float* vec_dat_ptr = mat_dat_ptr + mat_size_sum;

	size_t mat_accum_size = 0;
	size_t vec_accum_size = 0;
	for (int l = 0; l < ((int)layer_cnt-1); l++) {
		size_t l_size = nn->hidden_layers[l]->node_cnt;
		size_t l1_size = nn->hidden_layers[l+1]->node_cnt;

		matrix_init(l_size, l1_size, mat_dat_ptr + mat_accum_size, &mat_ptr[l]);
		mat_accum_size += matrix_calc_size(l_size, l1_size);

		vec_init(l1_size, vec_dat_ptr + vec_accum_size, &vec_ptr[l]);
		vec_accum_size += vec_calc_size(l1_size);
	}

	papool->biases = vec_ptr;
	papool->weights = mat_ptr;
	papool->activation_fn = act_fn_pptr;
	papool->activation_fn_d = act_fn_d_pptr;
	papool->base.layer_cnt = layer_cnt;
	papool->base.dptr = mat_dat_ptr;
	papool->base.data_size = (mat_size_sum + vec_size_sum) * sizeof(float);

	return papool;
}

FFNPropagationPool* _ffn_init_prpool(size_t layer_cnt, size_t padded_size_sum, FFNParams* nn) {
	FFNPropagationPool* ppool = (FFNPropagationPool*)allocate(sizeof(FFNPropagationPool));
	size_t vec_mdata_size = (layer_cnt * sizeof(Vector)) << 1;
	size_t padding = _pool_pad(vec_mdata_size);

	ppool->base.data = _pool_alloc(
			vec_mdata_size // Space for 2 set of layer vector data
			+ padding // Padding for alignment
			+ ((padded_size_sum * sizeof(float)) << 1) // Space for vector data
			);

	Vector* act_vec_ptr = (Vector*)(ppool->base.data);
	Vector* preact_vec_ptr = act_vec_ptr + layer_cnt;

	float* act_dat_ptr = (float*)((char*)(preact_vec_ptr + layer_cnt)+padding);
	float* preact_dat_ptr = act_dat_ptr + padded_size_sum;

	size_t accum_size = 0;
	for (int l = 0; l < (int)layer_cnt; l++) {
		size_t l_size = nn->hidden_layers[l]->node_cnt;
		vec_init(l_size, act_dat_ptr + accum_size, &act_vec_ptr[l]);
		vec_init(l_size, preact_dat_ptr + accum_size, &preact_vec_ptr[l]); // ffn_mempool.c:53

		accum_size += vec_calc_size(l_size);
	}

	ppool->activations = act_vec_ptr;
	ppool->preactivations = preact_vec_ptr;
	ppool->base.layer_cnt = layer_cnt;
	ppool->base.dptr = act_dat_ptr;
	ppool->base.data_size = (padded_size_sum * sizeof(float)) << 1;

	return ppool;
}

FFNGradientPool* _ffn_init_gpool(size_t layer_cnt, size_t mat_size_sum, size_t vec_size_sum, FFNParams* nn) {
	FFNGradientPool* gpool = (FFNGradientPool*)allocate(sizeof(FFNGradientPool));
	// ((layer_cnt-1) * sizeof(Matrix)) + ((layer_cnt+1) * sizeof(Vector));
	size_t mdata_size = (layer_cnt * (sizeof(Matrix) + sizeof(Vector))) - sizeof(Matrix);
	size_t padding = _pool_pad(mdata_size);

	gpool->base.data = _pool_alloc(
			mdata_size // Space for matrix + Vector pointer
			+ padding // Padding for alignment
			+ ((mat_size_sum + vec_size_sum) * sizeof(float)) // Space for matrix + vector data
			);

	Matrix* mat_ptr = (Matrix*)(gpool->base.data);
	Vector* vec_ptr = (Vector*)(mat_ptr + layer_cnt - 1);

	float* mat_dat_ptr = (float*)((char*)(vec_ptr + layer_cnt) + padding);
	float* vec_dat_ptr = mat_dat_ptr + mat_size_sum;

	size_t mat_accum_size = 0;
	size_t vec_accum_size = 0;
	for (int l = 0; l < ((int)layer_cnt-1); l++) {
		size_t l_size = nn->hidden_layers[l]->node_cnt;
		size_t l1_size = nn->hidden_layers[l+1]->node_cnt;

		matrix_init(l_size, l1_size, mat_dat_ptr + mat_accum_size, &mat_ptr[l]);
		mat_accum_size += matrix_calc_size(l_size, l1_size);

		vec_init(l_size, vec_dat_ptr + vec_accum_size, &vec_ptr[l]);
		vec_accum_size += vec_calc_size(l_size);
	}

	vec_init(nn->hidden_layers[layer_cnt-1]->node_cnt, vec_dat_ptr + vec_accum_size, &vec_ptr[layer_cnt-1]);

	gpool->gradient_b = vec_ptr;
	gpool->gradient_w = mat_ptr;
	gpool->base.layer_cnt = layer_cnt;
	gpool->base.dptr = mat_dat_ptr;
	gpool->base.data_size = (mat_size_sum + vec_size_sum) * sizeof(float);

	return gpool;
}

FFNIntermediatePool* _ffn_init_ipool(size_t layer_cnt, size_t mat_size_sum, size_t vec_size_sum, FFNParams* nn) {
	FFNIntermediatePool* ipool = (FFNIntermediatePool*)allocate(sizeof(FFNIntermediatePool));
	size_t mdata_size = (((layer_cnt-1) * sizeof(Matrix)) << 1) + ((layer_cnt+1) * sizeof(Vector));
	size_t padding = _pool_pad(mdata_size);

	ipool->base.data = _pool_alloc(
			mdata_size // Space Transpose matrices/activation derivative vector/error coefficient matrices pointers
			+ padding // Padding for alignment
			+ ((mat_size_sum * sizeof(float)) << 1) // Space for transpose/error coefficents data
			+ (vec_size_sum * sizeof(float)) // Space for activation derivative data
			);

	Matrix* mat_trsp_ptr = (Matrix*)(ipool->base.data);
	Matrix* err_coef_ptr = mat_trsp_ptr + layer_cnt - 1;
	Vector* a_deriv_ptr = (Vector*)(err_coef_ptr + layer_cnt -1);

	float* mat_trsp_dat_ptr = (float*)((char*)(a_deriv_ptr + layer_cnt+1) + padding);
	float* err_coef_dat_ptr = mat_trsp_dat_ptr + mat_size_sum;
	float* a_deriv_dat_ptr = err_coef_dat_ptr + mat_size_sum;

	size_t mat_accum_size = 0;
	size_t vec_accum_size = 0;
	for (int l = 0; l < ((int)layer_cnt-1); l++) {
		size_t l_size = nn->hidden_layers[l]->node_cnt;
		size_t l1_size = nn->hidden_layers[l+1]->node_cnt;

		matrix_init(l1_size, l_size, mat_trsp_dat_ptr + mat_accum_size, &mat_trsp_ptr[l]);
		matrix_init(l1_size, l_size, err_coef_dat_ptr + mat_accum_size, &err_coef_ptr[l]);
		mat_accum_size += matrix_calc_size(l1_size, l_size);

		vec_init(l_size, a_deriv_dat_ptr + vec_accum_size, &a_deriv_ptr[l]);
		vec_accum_size += vec_calc_size(l_size);
	}
	size_t Ln1_size = nn->hidden_layers[layer_cnt-1]->node_cnt;
	vec_init(Ln1_size, a_deriv_dat_ptr + vec_accum_size, &a_deriv_ptr[layer_cnt-1]);
	vec_init(Ln1_size, a_deriv_dat_ptr + vec_accum_size + vec_calc_size(Ln1_size), &a_deriv_ptr[layer_cnt]);

	ipool->weight_trsp = mat_trsp_ptr;
	ipool->err_coef = err_coef_ptr;
	ipool->a_deriv = a_deriv_ptr;
	ipool->base.layer_cnt = layer_cnt;
	ipool->base.dptr = mat_trsp_dat_ptr;
	ipool->base.data_size = ((mat_size_sum << 1) + vec_size_sum) * sizeof(float);

	return ipool;
}

FFNParameterPool* ffn_init_parameter_pool(FFNParams* nn) {
	size_t L = nn->hidden_size;
	LayerData** l_data = nn->hidden_layers;

	size_t mat_padded_size = 0;
	size_t vec_padded_size = 0;

	for (int l = 0; l < ((int)L-1); l++) {
		vec_padded_size += vec_calc_size(l_data[l+1]->node_cnt);
		mat_padded_size += matrix_calc_size(l_data[l]->node_cnt, l_data[l+1]->node_cnt);
	}

	return _ffn_init_papool(L, mat_padded_size, vec_padded_size, nn);
}
FFNPropagationPool* ffn_init_propagation_pool(FFNParams* nn) {
	size_t L = nn->hidden_size;
	LayerData** l_data = nn->hidden_layers;

	size_t total_padded_size = 0;
	for (int l = 0; l < (int)L; l++) {
		total_padded_size += vec_calc_size(l_data[l]->node_cnt);
	}

	return _ffn_init_prpool(L, total_padded_size, nn);
}
FFNGradientPool* ffn_init_gradient_pool(FFNParams* nn) {
	size_t L = nn->hidden_size;
	LayerData** l_data = nn->hidden_layers;

	size_t mat_padded_size = 0;
	size_t vec_padded_size = 0;

	for (int l = 0; l < (int)L; l++) {
		vec_padded_size += vec_calc_size(l_data[l]->node_cnt);
		if (l >= (int)L-1) continue;
		mat_padded_size += matrix_calc_size(l_data[l]->node_cnt, l_data[l+1]->node_cnt);
	}

	return _ffn_init_gpool(L, mat_padded_size, vec_padded_size, nn);
}
FFNIntermediatePool* ffn_init_intermediate_pool(FFNParams* nn) {
	size_t L = nn->hidden_size;
	LayerData** l_data = nn->hidden_layers;

	size_t mat_padded_size = 0;
	size_t vec_padded_size = 0;

	for (int l = 0; l < (int)L; l++) {
		vec_padded_size += vec_calc_size(l_data[l]->node_cnt);
		if (l >= (int)L-1) continue;
		mat_padded_size += matrix_calc_size(l_data[l+1]->node_cnt, l_data[l]->node_cnt);
	}

	return _ffn_init_ipool(L, mat_padded_size, vec_padded_size + l_data[L-1]->node_cnt, nn);
}

void ffn_deallocate_parameter_pool(FFNParameterPool* papool) {
	deallocate(papool->base.data);
	deallocate(papool);
}
void ffn_deallocate_propagation_pool(FFNPropagationPool* prpool) {
	deallocate(prpool->base.data);
	deallocate(prpool);
}
void ffn_deallocate_gradient_pool(FFNGradientPool* gpool) {
	deallocate(gpool->base.data);
	deallocate(gpool);
}
void ffn_deallocate_intermediate_pool(FFNIntermediatePool* ipool) {
	deallocate(ipool->base.data);
	deallocate(ipool);
}
