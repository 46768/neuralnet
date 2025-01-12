#include "ffn_init.h"

#include <stdlib.h>
#include <math.h>

#include "booltype.h"
#include "logger.h"
#include "allocator.h"
#include "random.h"

//////////////////////////////
// Parameter Initialization //
//////////////////////////////

float _ffn_he_init(size_t node_cnt) {
	float u1 = f_random(0.0f, 1.0f);
	float u2 = f_random(0.0f, 1.0f);

	float z0 = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);

	return z0 * sqrt(2.0 / node_cnt);
}

//////////////
// Creation //
//////////////

// Create a feed forward network
FFN* ffn_init() {
	FFN* ffn = (FFN*)calloc(1, sizeof(FFN));
	ffn->immutable = FALSE;

	return ffn;
}

//////////////////////////
// Layer Initialization //
//////////////////////////

// Push a dense (fully connected) layer
void ffn_init_dense(FFN* nn, size_t dense_size) {
	if (nn->immutable) {
		error("Unable to modify ffn: Immutable");
		return;
	}
	size_t hidden_size = nn->hidden_size;
	if (hidden_size <= nn->hidden_capacity) {
		nn->hidden_capacity += 10;
		nn->hidden_layers = realloc(nn->hidden_layers, nn->hidden_capacity*sizeof(size_t));		
		nn->weights = realloc(nn->weights, nn->hidden_capacity*sizeof(Matrix*));		
		nn->biases = realloc(nn->biases, nn->hidden_capacity*sizeof(Vector*));		
	}

	nn->hidden_layers[hidden_size] = dense_size;
	nn->hidden_size++;
}

// Finalize a network's layer
void ffn_init_params(FFN* nn) {
	size_t* layer_size = nn->hidden_layers;
	for (int i = 0; i < nn->hidden_size-1; i++) {
		size_t sx = layer_size[i];
		size_t sy = layer_size[i+1];

		//nn->weights[i] = matrix_rand(sx, sy, 0.0f, 1.0f);
		//nn->biases[i] = vec_rand(sy, 0.0f, 1.0f);
		nn->weights[i] = matrix_zero(sx, sy);
		nn->biases[i] = vec_zero(sy);

		for (int y = 0; y < sy; y++) {
			nn->biases[i]->data[y] = _ffn_he_init(sx);
			for (int x = 0; x < sx; x++) {
				nn->weights[i]->data[y*sx + x] = _ffn_he_init(sx);
			}
		}
	}

	nn->immutable = TRUE;
}

///////////////////////
// Memory Management //
///////////////////////

// Deallocate a network
void ffn_deallocate(FFN* nn) {
	for (int i = 0; i < nn->hidden_size-1; i++) {
		matrix_deallocate(nn->weights[i]);
		vec_deallocate(nn->biases[i]);
	}
	deallocate(nn->weights);
	deallocate(nn->biases);
	deallocate(nn->hidden_layers);
	deallocate(nn);
}
