#include "activation.h"

#include <math.h>

activation_fn resolve_activation_fn(ActivationFN fn_type) {
	switch (fn_type) {
		case ReLU:
			return nn_relu;
		case Sigmoid:
			return nn_sigmoid;
		default:
			return NULL;
	}
}

activation_fn_d resolve_activation_fn_d(ActivationFN fn_type) {
	switch (fn_type) {
		case ReLU:
			return nn_relu_d;
		case Sigmoid:
			return nn_sigmoid_d;
		default:
			return NULL;
	}
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
