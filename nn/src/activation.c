#include "activation.h"

#include <math.h>
#include <string.h>

#include "logger.h"

ActivationFn resolve_activation_fn(ActivationFNEnum fn_type) {
	switch (fn_type) {
		case ReLU:
			return nn_relu;
		case CReLU:
			return nn_crelu;
		case Sigmoid:
			return nn_sigmoid;
		case Softmax:
			return nn_softmax;
		case None:
			return nn_none_fn;
		case Logging:
			return nn_logging_fn;
		default:
			return NULL;
	}
}

ActivationFnD resolve_activation_fn_d(ActivationFNEnum fn_type) {
	switch (fn_type) {
		case ReLU:
			return nn_relu_d;
		case CReLU:
			return nn_crelu_d;
		case Sigmoid:
			return nn_sigmoid_d;
		case Softmax:
			return nn_none_fn_d;
		case None:
			return nn_none_fn_d;
		case Logging:
			return nn_none_fn_d;
		default:
			return NULL;
	}
}

char* resolve_activation_fn_str(ActivationFNEnum fn_type) {
	switch (fn_type) {
		case ReLU:
			return "ReLU";
		case Sigmoid:
			return "Sigmoid";
		case Softmax:
			return "Softmax";
		case None:
			return "None";
		case Logging:
			return "Logging";
		default:
			return "Unknown";
	}
}

static inline void _check_vec_size(Vector* a, Vector* b) {
#ifndef NO_BOUND_CHECK
	if (a->dimension != b->dimension) {
		fatal("Mismatched vector size, a: %zu b: %zu", a->dimension, b->dimension);
	}
#endif
}

//////////
// None //
//////////

void nn_none_fn(Vector* z, Vector* a) {
	_check_vec_size(z, a);
	memcpy(a->data, z->data, z->dimension*sizeof(float));
}
void nn_none_fn_d(Vector* z, Vector* d) {
	_check_vec_size(z, d);
	for (size_t i = 0; i < z->dimension; i++) {
		d->data[i] = 1.0f;
	}
}

//////////
// ReLU //
//////////

void nn_relu(Vector* z, Vector* a) {
	_check_vec_size(z, a);
	for (size_t i = 0; i < z->dimension; i++) {
		if (z->data[i] > 0) a->data[i] = z->data[i];
	}
}
void nn_relu_d(Vector* z, Vector* d) {
	_check_vec_size(z, d);
	for (size_t i = 0; i < z->dimension; i++) {
		if (z->data[i] > 0) {
			d->data[i] = 1.0f;
		} else {
			d->data[i] = 0.0f;
		}
	}
}
void nn_crelu(Vector* z, Vector* a) {
	_check_vec_size(z, a);
	for (size_t i = 0; i < z->dimension; i++) {
		if (z->data[i] > 0) a->data[i] = fminf(z->data[i], 1.0f);
	}
}
void nn_crelu_d(Vector* z, Vector* d) {
	_check_vec_size(z, d);
	for (size_t i = 0; i < z->dimension; i++) {
		if (0 > z->data[i] || z->data[i] > 1) {
			d->data[i] = 1.0f;
		} else {
			d->data[i] = 0.0f;
		}
	}
}

/////////////
// Sigmoid //
/////////////

static inline float _sigmoid(float x) { return (float)1/(1+expf(-x)); }
void nn_sigmoid(Vector* z, Vector* a) {
	_check_vec_size(z, a);
	for (size_t i = 0; i < z->dimension; i++) {
		a->data[i] = _sigmoid(z->data[i]);
	}
}
void nn_sigmoid_d(Vector* z, Vector* d) {
	_check_vec_size(z, d);
	for (size_t i = 0; i < z->dimension; i++) {
		float sig = _sigmoid(z->data[i]);
		d->data[i] = sig*(1-sig);
	}
}

/////////////
// Softmax //
/////////////

void nn_softmax(Vector* z, Vector* a) {
	_check_vec_size(z, a);
	// Find largest element of the vector
	float z_max = 0;
	for (size_t i = 0; i < z->dimension; i++) {
		if (z->data[i] > z_max) {
			z_max = z->data[i];
		}
	}

	float exp_sum = 0;
	for (size_t i = 0; i < z->dimension; i++) {
		float z_exp = exp(z->data[i] - z_max);
		exp_sum += z_exp;
		a->data[i] = z_exp;
	}
	for (size_t i = 0; i < z->dimension; i++) {
		a->data[i] /= exp_sum;
	}
}

/////////////
// Logging //
/////////////

void nn_logging_fn(Vector* z, Vector* a) {
	_check_vec_size(z, a);
	info("Layer Info:");
	for (size_t i = 0; i < z->dimension; i++) {
		printr("node[%zu]: %f\n", i, z->data[i]);
		a->data[i] = z->data[i];
	}
}
