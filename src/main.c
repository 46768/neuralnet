#include "logger.h"
#include "random.h"
#include "allocator.h"

#include "ffn_init.h"
#include "ffn_bpropagate.h"
#include "ffn_mempool.h"

#define REGS_RANGE 10
#define REGS_RANGEL 0

int main() {
	debug("init");
	init_random();
	// Inputs
	Vector** vecs = (Vector**)callocate(REGS_RANGE - REGS_RANGEL, sizeof(Vector*));
	Vector** targets = (Vector**)callocate(REGS_RANGE - REGS_RANGEL, sizeof(Vector*));
	for (int i = REGS_RANGEL; i < REGS_RANGE; i++) {
		vecs[i - REGS_RANGEL] = vec_zero(1); vecs[i - REGS_RANGEL]->data[0] = i;
		targets[i - REGS_RANGEL] = vec_zero(1); targets[i - REGS_RANGEL]->data[0] = 
			(4.0f*i) + 10.0f;
	}

	FFN* nn = ffn_init();
	ffn_init_dense(nn, 1, None);
	ffn_init_dense(nn, 1, None);
	ffn_set_cost_fn(nn, MSE);
	FFNMempool* mempool = ffn_init_pool(nn);
	ffn_init_params(nn);

	info("Pre train");
	info("Network weights:");
	for (size_t l = 0; l < nn->hidden_size-1; l++) {
		newline();
		Matrix* mat = nn->weights[l];
		info("layer %d: in:%zu out:%zu", l, mat->sx, mat->sy);
		for (size_t y = 0; y < mat->sy; y++) {
			for (size_t x = 0; x < mat->sx; x++) {
				printr("%f ", matrix_get(mat, x, y));
			}
			newline();
		}
	}
	newline();

	info("Network bias:");
	for (size_t l = 0; l < nn->hidden_size-1; l++) {
		newline();
		Vector* b = nn->biases[l];
		info("layer %d: %zu", l, b->dimension);
		for (size_t x = 0; x < b->dimension; x++) {
			printr("node %zu: %f\n", x, b->data[x]);
		}
		newline();
	}
	newline();

	float learning_rate = 0.01;
	// Trains 3 times
	for (int t = 0; t < 1000; t++) {
		float l = 0;
		//l += ffn_bpropagate(nn, vecs[1], targets[1], learning_rate);

		for (int i = REGS_RANGEL; i < REGS_RANGE; i++) {
			Vector* x = vecs[i - REGS_RANGEL];
			Vector* y = targets[i - REGS_RANGEL];
			l += ffn_bpropagate(nn, mempool, x, y, learning_rate);
			newline_d();
		}
		debug("Training loss: %f", l/REGS_RANGE - REGS_RANGEL);
		if ((l/(float)REGS_RANGE - REGS_RANGEL) <= 0.000000000001) {
			break;
		}

	}
	newline();
	info("Post train");
	info("Network weights:");
	for (size_t l = 0; l < nn->hidden_size-1; l++) {
		newline();
		Matrix* mat = nn->weights[l];
		info("layer %d: in:%zu out:%zu", l, mat->sx, mat->sy);
		for (size_t y = 0; y < mat->sy; y++) {
			for (size_t x = 0; x < mat->sx; x++) {
				printr("%f ", matrix_get(mat, x, y));
			}
			newline();
		}
	}
	newline();

	info("Network bias:");
	for (size_t l = 0; l < nn->hidden_size-1; l++) {
		newline();
		Vector* b = nn->biases[l];
		info("layer %d: %zu", l, b->dimension);
		for (size_t x = 0; x < b->dimension; x++) {
			printr("node %zu: %f\n", x, b->data[x]);
		}
		newline();
	}
	newline();

	for (int i = REGS_RANGEL; i < REGS_RANGE; i++) {
		vec_deallocate(vecs[i - REGS_RANGEL]);
		vec_deallocate(targets[i - REGS_RANGEL]);
	}
	deallocate(vecs);
	deallocate(targets);

	ffn_deallocate(nn);
	ffn_deallocate_pool(mempool);
	return 0;
}
