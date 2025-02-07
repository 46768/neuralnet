#include "ffn_mempool.h"

#include "ffn_type.h"

#include "vector.h"
#include "matrix.h"

#include "allocator.h"

FFNMempool* ffn_init_pool(FFN* nn) {
	FFNMempool* pool = (FFNMempool*)allocate(sizeof(FFNMempool));
	size_t L = nn->hidden_size;
	pool->layer_cnt = L;

	// Forward propagation
	pool->preactivations = (Vector**)callocate(L, sizeof(Vector*));
	pool->activations = (Vector**)callocate(L, sizeof(Vector*));

	// Backpropagation
	pool->gradient_b = (Vector**)callocate(L, sizeof(Vector*));
	pool->gradient_w = (Matrix**)callocate(L-1, sizeof(Matrix*));
	pool->gradient_aL_C = vec_zero(nn->hidden_layers[L-1]->node_cnt);
	pool->a_deriv_L = vec_zero(nn->hidden_layers[L-1]->node_cnt);

	pool->weight_trsp = (Matrix**)callocate(L-1, sizeof(Matrix*));
	pool->a_deriv = (Vector**)callocate(L-1, sizeof(Vector*));
	pool->err_coef = (Matrix**)callocate(L-1, sizeof(Matrix*));

	for (size_t l = 0; l < L; l++) {
		size_t l_size = nn->hidden_layers[l]->node_cnt;
		pool->preactivations[l] = vec_zero(l_size);
		pool->activations[l] = vec_zero(l_size);
		pool->gradient_b[l] = vec_zero(l_size);

		if (l >= L-1) continue; // Limit code below to l < L-1
		size_t l1_size = nn->hidden_layers[l+1]->node_cnt;

		pool->gradient_w[l] = matrix_zero(l_size, l1_size);
		pool->weight_trsp[l] = matrix_zero(l1_size, l_size);
		pool->a_deriv[l] = vec_zero(l_size);
		pool->err_coef[l] = matrix_zero(l1_size, l_size);
	}

	return pool;
}

void ffn_deallocate_pool(FFNMempool* pool) {
	size_t L = pool->layer_cnt;
	for (size_t l = 0; l < L; l++) {
		vec_deallocate(pool->preactivations[l]);
		vec_deallocate(pool->activations[l]);
		vec_deallocate(pool->gradient_b[l]);

		if (l >= L-1) continue; // Limit code below to l < L-1
		matrix_deallocate(pool->gradient_w[l]);
		matrix_deallocate(pool->weight_trsp[l]);
		vec_deallocate(pool->a_deriv[l]);
		matrix_deallocate(pool->err_coef[l]);
	}
	vec_deallocate(pool->gradient_aL_C);
	vec_deallocate(pool->a_deriv_L);

	deallocate(pool->preactivations);
	deallocate(pool->activations);
	deallocate(pool->gradient_w);
	deallocate(pool->gradient_b);
	deallocate(pool->weight_trsp);
	deallocate(pool->a_deriv);
	deallocate(pool->err_coef);

	deallocate(pool);
}
