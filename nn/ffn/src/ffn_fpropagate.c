#include "ffn_fpropagate.h"

#include <string.h>

#include "vector.h"
#include "matrix.h"

#include "logger.h"

void ffn_fpropagate(FFNParameterPool* papool, FFNPropagationPool* prpool, Vector* input) {
	Matrix* weights = papool->weights;
	Vector* biases = papool->biases;
	Vector* z = prpool->preactivations;
	Vector* a = prpool->activations;


	// Check if the input is compatible with network input
#ifndef NO_BOUND_CHECK
	if (input->dimension != (&weights[0])->sx) {
		fatal("Input vector size mismatched with input node count, %d to %d",
				input->dimension,
				(&weights[0])->sx
		);
		exit(1);
	}
#endif

	memcpy((&z[0])->data, input->data, input->dimension * sizeof(float));
	memcpy((&a[0])->data, input->data, input->dimension * sizeof(float));

	// Propagate input to each layer
	for (int l = 0; l < ((int)(papool->layer_cnt)-1); l++) {
		Matrix* weight = &weights[l];
		Vector* bias = &biases[l];
		Vector* al = &a[l];
		Vector* zl1 = &z[l+1];
		Vector* al1 = &a[l+1];

		// Calculate the matrix vector multiplication
		debug("w[%zu]*a[%zu]+b[%zu]", l, l, l);
		matrix_vec_mul_offset_ip(weight, al, bias, zl1);
		papool->activation_fn[l](zl1, al1);
	}
}
