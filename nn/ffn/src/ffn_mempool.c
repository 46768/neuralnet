#include "ffn_mempool.h"

#include "ffn_type.h"

#include "matrix.h"
#include "vector.h"

#include "allocator.h"

FFNMempool* ffn_init_pool(FFN* nn) {
	FFNMempool* pool = (FFNMempool*)allocate(sizeof(FFNMempool));
	pool->layer_cnt = nn->hidden_size;

	pool->preactivations = (Vector**)callocate(nn->hidden_size, sizeof(Vector*));
	pool->activations = (Vector**)callocate(nn->hidden_size, sizeof(Vector*));

	pool->gradient_b = (Vector**)callocate(nn->hidden_size, sizeof(Vector*));
	pool->gradient_w = (Matrix**)callocate(nn->hidden_size, sizeof(Matrix*));

	return pool;
}

void ffn_deallocate_pool(FFNMempool* pool) {
	
}
