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

	info("Pre train");
	info("Network weights:");
	for (int l = 0; l < nn->hidden_size-1; l++) {
		newline();
		Matrix* mat = nn->weights[l];
		info("layer %d: in:%zu out:%zu", l, mat->sx, mat->sy);
		for (int y = 0; y < mat->sy; y++) {
			for (int x = 0; x < mat->sx; x++) {
				printr("%f ", matrix_get(mat, x, y));
			}
			newline();
		}
	}
	newline();

	info("Network bias:");
	for (int l = 0; l < nn->hidden_size-1; l++) {
		newline();
		Vector* b = nn->biases[l];
		info("layer %d: %zu", l, b->dimension);
		for (int x = 0; x < b->dimension; x++) {
			printr("node %d: %f\n", x, b->data[x]);
		}
		newline();
	}
	newline();

	float learning_rate = 0.01;
	// Trains 3 times
	for (int t = 0; t < 3000; t++) {
		float l = 0;
		//l += ffn_bpropagate(nn, vecs[1], targets[1], learning_rate);
		
		for (int i = 0; i < 10; i++) {
			Vector* x = vec_zero(1); x->data[0] = i;
			Vector* y = vec_zero(1); y->data[0] = (4.0f*i) + 10.0f;
			l += ffn_bpropagate(nn, x, y, learning_rate);
			vec_deallocate(x);
			vec_deallocate(y);
		}
		info("Training loss: %f", l/10);
		if ((l/(float)10) <= 0.000000000001) {
			break;
		}

	}
	newline();
	info("Post train");
	info("Network weights:");
	for (int l = 0; l < nn->hidden_size-1; l++) {
		newline();
		Matrix* mat = nn->weights[l];
		info("layer %d: in:%zu out:%zu", l, mat->sx, mat->sy);
		for (int y = 0; y < mat->sy; y++) {
			for (int x = 0; x < mat->sx; x++) {
				printr("%f ", matrix_get(mat, x, y));
			}
			newline();
		}
	}
	newline();

	info("Network bias:");
	for (int l = 0; l < nn->hidden_size-1; l++) {
		newline();
		Vector* b = nn->biases[l];
		info("layer %d: %zu", l, b->dimension);
		for (int x = 0; x < b->dimension; x++) {
			printr("node %d: %f\n", x, b->data[x]);
		}
		newline();
	}
	newline();

	for (int i = 0; i < 4; i++) {
		vec_deallocate(vecs[i]);
		vec_deallocate(targets[i]);
	}
	deallocate(vecs);
	deallocate(targets);

	ffn_deallocate(nn);
	return 0;
}
