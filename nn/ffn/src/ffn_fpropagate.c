#include "ffn_fpropagate.h"

#include "logger.h"
#include "matrix.h"
#include "vector.h"

Vector* ffn_fpropagate(FFN* nn, Vector* input) {
	Matrix** weights = nn->weights;
	Vector** biases = nn->biases;

	// Check if the input is compatible with network input
	if (input->dimension != nn->hidden_layers[0]) {
		fatal("Input vector size mismatched with input node count, %d to %d",
				input->dimension,
				nn->hidden_layers[0]
		);
		exit(1);
	}
	Vector* output_activation = vec_dup(input);

	// Propagate input to each layer
	for (int l = 0; l < nn->hidden_size-1; l++) {
		Matrix* weight = weights[l];
		Vector* bias = biases[l];
		Vector* activation = vec_zero(bias->dimension);

		// Calculate the matrix vector multiplication
		for (int y = 0; y < weight->sy; y++) {
			float a_j = (bias->data)[y];
			for (int x = 0; x < weight->sx; x++) {
				a_j += (output_activation->data)[x]*matrix_get(weight, x, y);
			}
			(activation->data)[y] = a_j > 0 ? a_j : 0;
		}
		vec_deallocate(output_activation); // Deallocate previous layer activation
		output_activation = activation; // Override with the current layer activation
	}

	return output_activation;
}
