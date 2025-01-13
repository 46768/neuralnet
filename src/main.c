#include <stdio.h>

#include "logger.h"
#include "random.h"
#include "allocator.h"

#include "ffn_init.h"
#include "ffn_fpropagate.h"
#include "ffn_bpropagate.h"

int main() {
	init_random();
	// Inputs
	Vector** vecs = (Vector**)calloc(4, sizeof(Vector*));
	vecs[0] = vec_zero(2); vecs[0]->data[0] = -1.0f; vecs[0]->data[1] = -1.0f;
	vecs[1] = vec_zero(2); vecs[1]->data[0] = -1.0f; vecs[1]->data[1] = 1.0f;
	vecs[2] = vec_zero(2); vecs[2]->data[0] = 1.0f; vecs[2]->data[1] = -1.0f;
	vecs[3] = vec_zero(2); vecs[3]->data[0] = 1.0f; vecs[3]->data[1] = 1.0f;
	Vector** targets = (Vector**)calloc(4, sizeof(Vector*));
	targets[0] = vec_zero(1);
	targets[1] = vec_zero(1); targets[1]->data[0] = 1.0f;
	targets[2] = vec_zero(1); targets[2]->data[0] = 1.0f;
	targets[3] = vec_zero(1);

	FFN* nn = ffn_init();
	ffn_init_dense(nn, 1, None);
	ffn_init_dense(nn, 1, None);
	ffn_set_cost_fn(nn, MSE);
	ffn_init_params(nn);

	float learning_rate = 0.01;
	// Trains 3 times
	for (int t = 0; t < 1000; t++) {
		float l = 0;
		//l += ffn_bpropagate(nn, vecs[1], targets[1], learning_rate);
		
		for (int i = 0; i < 10; i++) {
			Vector* x = vec_zero(1); x->data[0] = i;
			Vector* y = vec_zero(1); x->data[0] = (5.0f*i) + 2.0f;
			l += ffn_bpropagate(nn, x, y, learning_rate);
		}
		if (t % 100 == 0) {
			debug("Training loss: %f", l/4);
		}
	}
	newline();
	info("Post train");
	debug("Network weights:");
	for (int l = 0; l < nn->hidden_size-1; l++) {
		newline_d();
		Matrix* mat = nn->weights[l];
		debug("layer %d: in:%zu out:%zu", l, mat->sx, mat->sy);
		for (int y = 0; y < mat->sy; y++) {
			for (int x = 0; x < mat->sx; x++) {
				printr_d("%f ", matrix_get(mat, x, y));
			}
			newline_d();
		}
	}
	newline_d();

	debug("Network bias:");
	for (int l = 0; l < nn->hidden_size-1; l++) {
		newline_d();
		Vector* b = nn->biases[l];
		debug("layer %d: %zu", l, b->dimension);
		for (int x = 0; x < b->dimension; x++) {
			printr_d("node %d: %f\n", x, b->data[x]);
		}
		newline_d();
	}
	newline_d();

	for (int i = 0; i < 4; i++) {
		vec_deallocate(vecs[i]);
		vec_deallocate(targets[i]);
	}
	deallocate(vecs);
	deallocate(targets);

	ffn_deallocate(nn);
	return 0;
}
