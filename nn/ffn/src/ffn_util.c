#include "ffn_util.h"

#include "logger.h"

void ffn_dump_data(FFN* nn) {
	info("Network weights:");
	for (size_t l = 0; l < nn->hidden_size-1; l++) {
		newline();
		Matrix* mat = nn->weights[l];
		info("layer %d: in:%zu out:%zu", l, mat->sx, mat->sy);
		for (size_t y = 0; y < mat->sy; y++) {
			for (size_t x = 0; x < mat->sx; x++) {
				printr("%f ", matrix_get(mat, x, y));
			}
			newline();
		}
	}
	newline();

	info("Network bias:");
	for (size_t l = 0; l < nn->hidden_size-1; l++) {
		newline();
		Vector* b = nn->biases[l];
		info("layer %d: %zu", l, b->dimension);
		for (size_t x = 0; x < b->dimension; x++) {
			printr("node %zu: %f\n", x, b->data[x]);
		}
		newline();
	}
	newline();
}
