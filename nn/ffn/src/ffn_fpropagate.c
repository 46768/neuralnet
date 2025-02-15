#include "ffn_fpropagate.h"

#include "vector.h"
#include "matrix.h"

#include "logger.h"

void ffn_fpropagate(FFN* nn, FFNPropagationPool* pool, Vector* input) {
	if (!(nn->immutable)) {
		fatal("Unable to forward propagate: Network is mutable");
	}

	Matrix** weights = nn->weights;
	Vector** biases = nn->biases;
	Vector* z = pool->preactivations;
	Vector* a = pool->activations;


	// Check if the input is compatible with network input
	if (input->dimension != nn->hidden_layers[0]->node_cnt) {
		fatal("Input vector size mismatched with input node count, %d to %d",
				input->dimension,
				nn->hidden_layers[0]
		);
		exit(1);
	}

	for (int i = 0; i < (int)(input->dimension); i++) {
		z[0].data[i] = input->data[i];
		a[0].data[i] = input->data[i];
	}

	// Propagate input to each layer
	for (size_t l = 0; l < nn->hidden_size-1; l++) {
		Matrix* weight = weights[l];
		Vector* bias = biases[l];
		Vector* al = &a[l];
		Vector* zl1 = &z[l+1];
		Vector* al1 = &a[l+1];

		// Calculate the matrix vector multiplication
		debug("w[%zu]*a[%zu]+b[%zu]", l, l, l);
		matrix_vec_mul_offset_ip(weight, al, bias, zl1);
		nn->layer_activation[l](zl1, al1);
	}
}
