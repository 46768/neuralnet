#include "ffn_util.h"

#include "logger.h"

void _dump_vector(Vector* vec) {
	for (int i = 0; i < (int)(vec->dimension); i++) {
		printr("%.10f\n", vec->data[i]);
	}
}

void _dump_matrix(Matrix* mat) {
	for (size_t y = 0; y < mat->sy; y++) {
		for (size_t x = 0; x < mat->sx; x++) {
			printr("%f ", matrix_get(mat, x, y));
		}
		newline();
	}
}

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

void ffn_dump_pool(FFNMempool* pool) {
	size_t L = pool->layer_cnt;
	info("Pool Layer Count: %zu", L);
	info("Pool Preactivations:");
	for (int l = 0; l < (int)L; l++) {
		_dump_vector(&(pool->propagation->preactivations[l]));
		newline();
	}

	info("Pool Activations:");
	for (int l = 0; l < (int)L; l++) {
		_dump_vector(&(pool->propagation->activations[l]));
		newline();
	}

	info("Pool Bias Gradient:");
	for (int l = 0; l < (int)L; l++) {
		_dump_vector(&(pool->gradients->gradient_b[l]));
		newline();
	}

	info("Pool Weight Gradient:");
	for (int l = 0; l < (int)L-1; l++) {
		_dump_matrix(&(pool->gradients->gradient_w[l]));
		newline();
	}

	info("Pool Cost Gradient:");
	_dump_vector(&(pool->intermediates->a_deriv[L]));
	newline();
	info("Pool L Derivative:");
	_dump_vector(&(pool->intermediates->a_deriv[L-1]));
	newline();

	info("Pool Weight Transpose");
	for (int l = 0; l < (int)L-1; l++) {
		_dump_matrix(&(pool->intermediates->weight_trsp[l]));
		newline();
	}
	info("Pool Layer Derivative");
	for (int l = 0; l < (int)L-1; l++) {
		_dump_vector(&(pool->intermediates->a_deriv[l]));
		newline();
	}
	info("Pool Error Coefficient");
	for (int l = 0; l < (int)L-1; l++) {
		_dump_matrix(&(pool->intermediates->err_coef[l]));
		newline();
	}
}

void ffn_dump_output(FFNMempool* pool) {
	size_t layer_count = pool->layer_cnt;
	Vector* output = &(pool->propagation->activations[layer_count-1]);

	info("Network output:");
	for (size_t i = 0; i < output->dimension; i++) {
		printr("%f\n", output->data[i]);
	}
	newline();
}
