#include <math.h>

#include "logger.h"
#include "random.h"
#include "allocator.h"

#include "generator.h"

#include "grapher.h"

#include "ffn_init.h"
#include "ffn_fpropagate.h"
#include "ffn_bpropagate.h"
#include "ffn_mempool.h"
#include "ffn_util.h"

int main() {
	debug("init");
	init_random();
	// Inputs
	Vector** vecs;
	Vector** targets;
	int REGS_RANGE = 10;
	int REGS_RANGEL = -10;
	generate_linear_regs(REGS_RANGEL, REGS_RANGE, 4.0f, -5.0f, &vecs, &targets);
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
	// Trains however many times
	for (int t = 0; t < 1000; t++) {
		float l = 0;
		for (int i = REGS_RANGEL; i <= REGS_RANGE; i++) {
			Vector* x = vecs[i - REGS_RANGEL];
			Vector* y = targets[i - REGS_RANGEL];
			l += ffn_bpropagate(nn, mempool, x, y, learning_rate);
			newline_d();
		}
		l /= (REGS_RANGE - REGS_RANGEL);
		if (t % 1 == 0) {
			info("Training loss: %f", l);
		}
	}

	info("Post train");
	ffn_dump_data(nn);

	for (int i = REGS_RANGEL; i <= REGS_RANGE; i++) {
		vec_deallocate(vecs[i - REGS_RANGEL]);
		vec_deallocate(targets[i - REGS_RANGEL]);
	}
	deallocate(vecs);
	deallocate(targets);

	ffn_deallocate(nn);
	ffn_deallocate_pool(mempool);
	python_graph("hello");
	return 0;
}
