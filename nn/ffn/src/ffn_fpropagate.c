#include "ffn_fpropagate.h"

#include <string.h>

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
	memcpy(z[0]->data, input->data, input->dimension*sizeof(float));
	memcpy(a[0]->data, input->data, input->dimension*sizeof(float));

	// Propagate input to each layer
	for (int l = 0; l < nn->hidden_size-1; l++) {
		Matrix* weight = weights[l];
		Vector* bias = biases[l];
		Vector* zl = z[l];
		Vector* zl1 = z[l+1];
		Vector* al = a[l+1];

		// Calculate the matrix vector multiplication
		for (int y = 0; y < weight->sy; y++) {
			float a_j = (bias->data)[y];
			for (int x = 0; x < weight->sx; x++) {
				a_j += (zl->data)[x]*matrix_get(weight, x, y);
			}
			(zl1->data)[y] = a_j;
		}
		nn->layer_activation[l](zl, al);
	}
}
