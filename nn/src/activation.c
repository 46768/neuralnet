#include "activation.h"

#include <math.h>
#include <string.h>

#include "logger.h"

ActivationFn resolve_activation_fn(ActivationFNEnum fn_type) {
	switch (fn_type) {
		case ReLU:
			return nn_relu;
		case Sigmoid:
			return nn_sigmoid;
		case None:
			return nn_none_fn;
		default:
			return NULL;
	}
}

ActivationFnD resolve_activation_fn_d(ActivationFNEnum fn_type) {
	switch (fn_type) {
		case ReLU:
			return nn_relu_d;
		case Sigmoid:
			return nn_sigmoid_d;
		case None:
			return nn_none_fn_d;
		default:
			return NULL;
	}
}

//////////
// None //
//////////

void nn_none_fn(Vector* z, Vector* a) {
	if (z->dimension != a->dimension) {
		fatal("Mismatched vector size, z: %zu a: %zu", z->dimension, a->dimension);
	}
	memcpy(a->data, z->data, z->dimension*sizeof(float));
}
void nn_none_fn_d(Vector* z, Vector* d) {
	if (z->dimension != d->dimension) {
		fatal("Mismatched vector size, z: %zu d: %zu", z->dimension, d->dimension);
	}
	for (int i = 0; i < z->dimension; i++) {
		d->data[i] = 1.0f;
	}
}

//////////
// ReLU //
//////////

void nn_relu(Vector* z, Vector* a) {
	if (z->dimension != a->dimension) {
		fatal("Mismatched vector size, z: %zu a: %zu", z->dimension, a->dimension);
	}
	for (int i = 0; i < z->dimension; i++) {
		if (z->data[i] > 0) a->data[i] = z->data[i];
	}
}
void nn_relu_d(Vector* z, Vector* d) {
	if (z->dimension != d->dimension) {
		fatal("Mismatched vector size, z: %zu d: %zu", z->dimension, d->dimension);
	}
	for (int i = 0; i < z->dimension; i++) {
		if (z->data[i] > 0) d->data[i] = 1.0f;
	}
}

/////////////
// Sigmoid //
/////////////

static inline float _sigmoid(float x) { return (float)1/(1+exp(-x)); }
void nn_sigmoid(Vector* z, Vector* a) {
	if (z->dimension != a->dimension) {
		fatal("Mismatched vector size, z: %zu a: %zu", z->dimension, a->dimension);
	}
	for (int i = 0; i < z->dimension; i++) {
		a->data[i] = _sigmoid(z->data[i]);
	}
}
void nn_sigmoid_d(Vector* z, Vector* d) {
	if (z->dimension != d->dimension) {
		fatal("Mismatched vector size, z: %zu d: %zu", z->dimension, d->dimension);
	}
	for (int i = 0; i < z->dimension; i++) {
		float sig = _sigmoid(z->data[i]);
		d->data[i] = sig*(1-sig);
	}
}

/////////////
// Softmax //
/////////////

void nn_softmax(Vector* z, Vector* a) {
	if (z->dimension == 1) {
		return nn_sigmoid(z, a);
	}
	float exp_sum = 0;
	float z_max = 0;
	for (int i = 0; i < z->dimension; i++) {
		if (z->data[i] > z_max) {
			z_max = z->data[i];
		}
	}
	for (int i = 0; i < z->dimension; i++) {
		float z_exp = exp(z->data[i] - z_max);
		exp_sum += exp(z->data[i] - z_max);
		a->data[i] = z_exp;
	}
	for (int i = 0; i < z->dimension; i++) {
		a->data[i] /= exp_sum;
	}
}
