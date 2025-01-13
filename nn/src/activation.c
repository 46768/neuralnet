#include "activation.h"

#include <math.h>
#include <string.h>

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

Vector* nn_none_fn(Vector* z) {
	return vec_dup(z);
}
Vector* nn_none_fn_d(Vector* z) {
	Vector* a = vec_zero(z->dimension);
	memset(a->data, 1, a->dimension*sizeof(float));
	return a;
}

//////////
// ReLU //
//////////

Vector* nn_relu(Vector* z) {
	Vector* a = vec_zero(z->dimension);
	for (int i = 0; i < z->dimension; i++) {
		if (z->data[i] > 0) a->data[i] = z->data[i];
	}
	return a;
}
Vector* nn_relu_d(Vector* z) {
	Vector* d = vec_zero(z->dimension);
	for (int i = 0; i < z->dimension; i++) {
		if (z->data[i] > 0) d->data[i] = 1.0f;
	}
	return d;
}

/////////////
// Sigmoid //
/////////////

static inline float _sigmoid(float x) { return (float)1/(1+exp(-x)); }
Vector* nn_sigmoid(Vector* z) {
	Vector* a = vec_zero(z->dimension);
	for (int i = 0; i < z->dimension; i++) {
		a->data[i] = _sigmoid(z->data[i]);
	}
	return a;
}
Vector* nn_sigmoid_d(Vector* z) {
	Vector* d = vec_zero(z->dimension);
	for (int i = 0; i < z->dimension; i++) {
		float sig = _sigmoid(z->data[i]);
		d->data[i] = sig*(1-sig);
	}
	return d;
}

/////////////
// Softmax //
/////////////

Vector* nn_softmax(Vector* z) {
	if (z->dimension == 1) {
		return nn_sigmoid(z);
	}
	float exp_sum = 0;
	float z_max = 0;
	for (int i = 0; i < z->dimension; i++) {
		if (z->data[i] > z_max) {
			z_max = z->data[i];
		}
	}
	Vector* a = vec_zero(z->dimension);
	for (int i = 0; i < z->dimension; i++) {
		float z_exp = exp(z->data[i] - z_max);
		exp_sum += exp(z->data[i] - z_max);
		a->data[i] = z_exp;
	}
	for (int i = 0; i < z->dimension; i++) {
		a->data[i] /= exp_sum;
	}
	return a;
}
