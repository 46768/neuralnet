#include "ffn_fpropagate.h"

#include "logger.h"
#include "matrix.h"
#include "vector.h"

void ffn_fpropagate(FFN* nn, FFNMempool* pool, Vector* input) {
	if (!(nn->immutable)) {
		fatal("Unable to forward propagate: Network is mutable");
	}

	Matrix** weights = nn->weights;
	Vector** biases = nn->biases;
	Vector** z = pool->preactivations;
	Vector** a = pool->activations;


	// Check if the input is compatible with network input
	if (input->dimension != nn->hidden_layers[0]->node_cnt) {
		fatal("Input vector size mismatched with input node count, %d to %d",
				input->dimension,
				nn->hidden_layers[0]
		);
		exit(1);
	}
	for (size_t i = 0; i < input->dimension; i++) {
		z[0]->data[i] = input->data[i];
		a[0]->data[i] = input->data[i];
	}

	// Propagate input to each layer
	for (size_t l = 0; l < nn->hidden_size-1; l++) {
		Matrix* weight = weights[l];
		Vector* bias = biases[l];
		Vector* al = a[l];
		Vector* zl1 = z[l+1];
		Vector* al1 = a[l+1];

		// Calculate the matrix vector multiplication
		for (size_t y = 0; y < weight->sy; y++) {
			float z_j = (bias->data)[y];
			for (size_t x = 0; x < weight->sx; x++) {
				z_j += (al->data)[x]*matrix_get(weight, x, y);
				//debug("w: %f", matrix_get(weight, x, y));
			}
			(zl1->data)[y] = z_j;
		}
		nn->layer_activation[l](zl1, al1);
	}
}
