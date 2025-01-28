#include <math.h>

#include "logger.h"
#include "random.h"
#include "allocator.h"

#include "ffn_init.h"
#include "ffn_fpropagate.h"
#include "ffn_bpropagate.h"
#include "ffn_mempool.h"
#include "ffn_util.h"

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
	int REGS_RANGEL = -10;
	generate_linear_regs(REGS_RANGEL, REGS_RANGE, 4.0f, 5.0f, &vecs, &targets);
	//generate_xor(&REGS_RANGEL, &REGS_RANGE, &vecs, &targets);

	FFN* nn = ffn_init();
	ffn_init_dense(nn, 1, None, He, RandomEN2);
	ffn_init_dense(nn, 1, None, He, RandomEN2);
	ffn_set_cost_fn(nn, MSE);
	FFNMempool* mempool = ffn_init_pool(nn);
	ffn_init_params(nn);

	info("Pre train");
	ffn_dump_data(nn);

	float learning_rate = 0.01;
	float threshold = 0.0000000001;
	float prev_l = INFINITY;
	// Trains however many times
	for (int t = 0; t < 1; t++) {
		float l = 0;
		for (int i = REGS_RANGEL; i < REGS_RANGE; i++) {
			Vector* x = vecs[i - REGS_RANGEL];
			Vector* y = targets[i - REGS_RANGEL];
			l += ffn_bpropagate(nn, mempool, x, y, learning_rate);
			newline_d();
		}
		l /= (REGS_RANGE - REGS_RANGEL);
		if (fabsf(prev_l-l) <= threshold && l <= threshold) {
			info("Loss doestn decrease much");
			break;
		}
		prev_l = l;
		if (t % 1 == 0) {
			info("Training loss: %f", l);
		}
	}

	info("Post train");
	ffn_dump_data(nn);

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
