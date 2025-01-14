#include "ffn_mempool.h"

#include "ffn_type.h"

#include "matrix.h"
#include "vector.h"

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
	pool->gradient_w = (Matrix**)callocate(L, sizeof(Matrix*));
	pool->gradient_aL_C = vec_zero(nn->hidden_layers[L-1]->node_cnt);
	pool->a_deriv_L = vec_zero(nn->hidden_layers[L-1]->node_cnt);

	pool->weight_trsp = (Matrix**)callocate(L, sizeof(Matrix*));
	pool->a_deriv = (Vector**)callocate(L, sizeof(Vector*));
	pool->err_coef = (Matrix**)callocate(L, sizeof(Matrix*));

	return pool;
}

void ffn_deallocate_pool(FFNMempool* pool) {
	
}
