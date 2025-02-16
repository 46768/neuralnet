#include "ffn_mempool.h"

#include "ffn_type.h"

#include "vector.h"
#include "matrix.h"

#include "allocator.h"

#ifdef SIMD_AVX
#include "avx.h"
#define _pool_alloc(s) avx_allocate(s)
#define _pool_pad(s) ((s+63)&~63)-s
#else
#define _pool_alloc(s) callocate(s)
#define _pool_pad(s) 0
#endif

/**
 * \brief Allocate a propagation pool with known layer size
 *
 * Allocates a forward propagation memory pool with known layer count, activation size and preactivation size sums
 *
 * \param layer_cnt Layer count in the network
 * \param padded_size_sum Sum of all layer's size in the network padded with vec_get_size
 * \param nn Feedforward network data
 *
 * \return A memory pool for forward propagation activations
 *
 * \since 0.0.1.1
 */
FFNPropagationPool* _ffn_init_ppool(size_t layer_cnt, size_t padded_size_sum, FFN* nn) {
	FFNPropagationPool* ppool = (FFNPropagationPool*)allocate(sizeof(FFNPropagationPool));
	size_t vec_mdata_size = (layer_cnt * sizeof(Vector)) << 1;
	size_t padding = _pool_pad(vec_mdata_size);

	ppool->data = _pool_alloc(
			vec_mdata_size // Space for 2 set of layer vector data
			+ padding // Padding for alignment
			+ ((padded_size_sum * sizeof(float)) << 1) // Space for vector data
			);

	Vector* act_vec_ptr = (Vector*)(ppool->data);
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

	return ppool;
}

/**
 * \brief Allocate a gradient pool with known layer size
 *
 * Allocates a gradient memory pool with known layer count, weight matrix size and bias size sum
 *
 * \param layer_cnt Layer count in the network
 * \param mat_size_sum Sum of all layer's weight matrix size (padded with matrix_get_size)
 * \param vec_size_sum Sum of all layer's bias vector size (padded with vec_get_size)
 * \param nn Feedforward network data
 *
 * \return A memory pool for gradient values
 *
 * \since 0.0.1.1
 */
FFNGradientPool* _ffn_init_gpool(size_t layer_cnt, size_t mat_size_sum, size_t vec_size_sum, FFN* nn) {
	FFNGradientPool* gpool = (FFNGradientPool*)allocate(sizeof(FFNGradientPool));
	size_t mdata_size = ((layer_cnt-1) * sizeof(Matrix)) + (layer_cnt * sizeof(Vector));
	size_t padding = _pool_pad(mdata_size);

	gpool->data = _pool_alloc(
			mdata_size // Space for matrix + Vector pointer
			+ padding // Padding for alignment
			+ ((mat_size_sum + vec_size_sum) * sizeof(float)) // Space for matrix + vector data
			);

	Matrix* mat_ptr = (Matrix*)(gpool->data);
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

	return gpool;
}

/**
 * \brief Allocate an intermediate pool with known layer size
 *
 * Allocates a intermediate memory pool with known layer count and sizes
 *
 * \param layer_cnt Layer count in the network
 * \param mat_size_sum Sum of all layer's weight matrix size (padded with matrix_get_size)
 * \param vec_size_sum Sum of all layer's bias vector size (padded with vec_get_size)
 * \param nn Feedforward network data
 *
 * \return A memory pool for holding intermediates
 *
 * \since 0.0.1.1
 */
FFNIntermediatePool* _ffn_init_ipool(size_t layer_cnt, size_t mat_size_sum, size_t vec_size_sum, FFN* nn) {
	FFNIntermediatePool* ipool = (FFNIntermediatePool*)allocate(sizeof(FFNIntermediatePool));
	size_t mdata_size = (((layer_cnt-1) * sizeof(Matrix)) << 1) + ((layer_cnt+1) * sizeof(Vector));
	size_t padding = _pool_pad(mdata_size);

	ipool->data = _pool_alloc(
			mdata_size // Space Transpose matrices/activation derivative vector/error coefficient matrices pointers
			+ padding // Padding for alignment
			+ ((mat_size_sum * sizeof(float)) << 1) // Space for transpose/error coefficents data
			+ (vec_size_sum * sizeof(float)) // Space for activation derivative data
			);

	Matrix* mat_trsp_ptr = (Matrix*)(ipool->data);
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

	return ipool;
}

FFNPropagationPool* ffn_init_propagation_pool(FFN* nn) {
	size_t L = nn->hidden_size;
	LayerData** l_data = nn->hidden_layers;

	size_t total_padded_size = 0;
	for (int l = 0; l < (int)L; l++) {
		total_padded_size += vec_calc_size(l_data[l]->node_cnt);
	}

	return _ffn_init_ppool(L, total_padded_size, nn);
}
FFNGradientPool* ffn_init_gradient_pool(FFN* nn) {
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
FFNIntermediatePool* ffn_init_intermediate_pool(FFN* nn) {
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

FFNMempool* ffn_init_pool(FFN* nn) {
	FFNMempool* pool = (FFNMempool*)allocate(sizeof(FFNMempool));
	size_t L = nn->hidden_size;
	LayerData** l_data = nn->hidden_layers;
	pool->layer_cnt = L;

	size_t mat_padded_size = 0;
	size_t vec_padded_size = 0;

	for (int l = 0; l < (int)L; l++) {
		vec_padded_size += vec_calc_size(l_data[l]->node_cnt);
		if (l >= (int)L-1) continue;
		mat_padded_size += matrix_calc_size(l_data[l+1]->node_cnt, l_data[l]->node_cnt);
	}

	pool->propagation = _ffn_init_ppool(L, vec_padded_size, nn);
	pool->gradients = _ffn_init_gpool(L, mat_padded_size, vec_padded_size, nn);
	pool->intermediates = _ffn_init_ipool(L, mat_padded_size, vec_padded_size + vec_calc_size(l_data[L-1]->node_cnt), nn);

	return pool;
}

void ffn_deallocate_propagation_pool(FFNPropagationPool* ppool) {
	deallocate(ppool->data);
	deallocate(ppool);
}
void ffn_deallocate_gradient_pool(FFNGradientPool* gpool) {
	deallocate(gpool->data);
	deallocate(gpool);
}
void ffn_deallocate_intermediate_pool(FFNIntermediatePool* ipool) {
	deallocate(ipool->data);
	deallocate(ipool);
}
void ffn_deallocate_pool(FFNMempool* pool) {
	ffn_deallocate_propagation_pool(pool->propagation);
	ffn_deallocate_gradient_pool(pool->gradients);
	ffn_deallocate_intermediate_pool(pool->intermediates);
	deallocate(pool);
}
