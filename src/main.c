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
	vecs[0] = vec_zero(2);
	vecs[1] = vec_zero(2); vecs[1]->data[0] = 1.0f;
	vecs[2] = vec_zero(2); vecs[2]->data[1] = 1.0f;
	vecs[3] = vec_zero(2); vecs[3]->data[0] = 1.0f; vecs[0]->data[1] = 1.0f;
	Vector** targets = (Vector**)calloc(4, sizeof(Vector*));
	targets[0] = vec_zero(2); targets[0]->data[1] = 1.0f;
	targets[1] = vec_zero(2); targets[1]->data[0] = 1.0f;
	targets[2] = vec_zero(2); targets[2]->data[0] = 1.0f;
	targets[3] = vec_zero(2); targets[3]->data[1] = 1.0f;

	FFN* nn = ffn_init();
	ffn_init_dense(nn, 2);
	ffn_init_dense(nn, 16);
	ffn_init_dense(nn, 2);
	ffn_init_params(nn);

	Vector* outpre = ffn_fpropagate(nn, vecs[1]);
	for (int i = 0; i < outpre->dimension; i++) {
		info("out[%d]: %f", i, outpre->data[i]);
	}
	vec_deallocate(outpre);

	float learning_rate = 0.01;
	// Trains 3 times
	for (int t = 0; t < 30; t++) {
		for (int i = 0; i < 4; i++) {
			ffn_bpropagate(nn, vecs[i], targets[i], learning_rate);
		}
		Vector* outt = ffn_fpropagate(nn, vecs[1]);
		for (int i = 0; i < outt->dimension; i++) {
			info("out[%d]: %f", i, outt->data[i]);
		}
		newline();
		vec_deallocate(outt);
	}
	info("Post train");
	for (int i = 0; i < 4; i++) {
		Vector* outt = ffn_fpropagate(nn, vecs[i]);
		for (int i = 0; i < outt->dimension; i++) {
			info("out[%d]: %f", i, outt->data[i]);
		}
		newline();
		vec_deallocate(outt);
	}

	for (int i = 0; i < 4; i++) {
		vec_deallocate(vecs[i]);
		vec_deallocate(targets[i]);
	}
	deallocate(vecs);
	deallocate(targets);

	ffn_deallocate(nn);
	return 0;
}
