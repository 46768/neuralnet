#include "logger.h"
#include "random.h"
#include "allocator.h"

#include "ffn_init.h"
#include "ffn_fpropagate.h"
#include "ffn_bpropagate.h"
#include "ffn_mempool.h"

void generate_linear_regs(int lower, int upper, float m, float y, Vector*** vecs, Vector*** targets) {
	*vecs = (Vector**)callocate(upper - lower, sizeof(Vector*));
	*targets = (Vector**)callocate(upper - lower, sizeof(Vector*));
	for (int i = lower; i < upper; i++) {
		(*vecs)[i - lower] = vec_zero(1); (*vecs)[i - lower]->data[0] = i;
		(*targets)[i - lower] = vec_zero(1); (*targets)[i - lower]->data[0] = (m*i) + y;
	}
}

void generate_xor(int* lower, int* upper, Vector*** vecs, Vector*** targets) {
	*lower = 0;
	*upper = 4;
	*vecs = (Vector**)callocate(*upper - *lower, sizeof(Vector*));
	*targets = (Vector**)callocate(*upper - *lower, sizeof(Vector*));

	(*vecs)[0] = vec_zero(2);(*vecs)[0]->data[0]=-1.0f;(*vecs)[0]->data[1]=-1.0f;
	(*vecs)[1] = vec_zero(2);(*vecs)[1]->data[0]=-1.0f;(*vecs)[1]->data[1]=1.0f;
	(*vecs)[2] = vec_zero(2);(*vecs)[2]->data[0]=1.0f;(*vecs)[2]->data[1]=-1.0f;
	(*vecs)[3] = vec_zero(2);(*vecs)[3]->data[0]=1.0f;(*vecs)[3]->data[1]=1.0f;

	(*targets)[0] = vec_zero(1);(*targets)[0]->data[0] = 0.0f;
	(*targets)[1] = vec_zero(1);(*targets)[1]->data[0] = 1.0f;
	(*targets)[2] = vec_zero(1);(*targets)[2]->data[0] = 1.0f;
	(*targets)[3] = vec_zero(1);(*targets)[3]->data[0] = 0.0f;
}

int main() {
	debug("init");
	init_random();
	// Inputs
	Vector** vecs;
	Vector** targets;
	int REGS_RANGE = 10;
	int REGS_RANGEL = 0;
	//generate_linear_regs(REGS_RANGEL, REGS_RANGE, 4.0f, 5.0f, &vecs, &targets);
	generate_xor(&REGS_RANGEL, &REGS_RANGE, &vecs, &targets);

	FFN* nn = ffn_init();
	ffn_init_dense(nn, 2, ReLU);
	ffn_init_dense(nn, 4, Sigmoid);
	ffn_init_dense(nn, 1, None);
	ffn_set_cost_fn(nn, CEL);
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
	for (int t = 0; t < 1000000; t++) {
		float l = 0;
		for (int i = REGS_RANGEL; i < REGS_RANGE; i++) {
			Vector* x = vecs[i - REGS_RANGEL];
			Vector* y = targets[i - REGS_RANGEL];
			l += ffn_bpropagate(nn, mempool, x, y, learning_rate);
			newline_d();
		}
		l /= (REGS_RANGE - REGS_RANGEL);
		if (t % 100 == 0) {
			info("Training loss: %f", l);
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
	info("Testing:");
	for (int i = REGS_RANGEL; i < REGS_RANGE; i++) {
		Vector* x = vecs[i - REGS_RANGEL];
		ffn_fpropagate(nn, mempool, x);
		info("set %d", i);
		info("In:");
		Vector* a = mempool->activations[0];
		for (size_t x = 0; x < a->dimension; x++) {
			printr("node %zu: %f\n", x, a->data[x]);
		}
		info("Out:");
		Vector* o = mempool->activations[nn->hidden_size-1];
		for (size_t x = 0; x < o->dimension; x++) {
			printr("node %zu: %f\n", x, o->data[x]);
		}
		newline();

		newline();
	}

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
