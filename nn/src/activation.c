#include "activation.h"

#include <math.h>
#include <string.h>

#ifdef SIMD_AVX
#include "avxmm.h"
#endif

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

//////////
// None //
//////////

void nn_none_fn(Vector* z, Vector* a) {
#ifndef NO_BOUND_CHECK
	if (z->dimension != a->dimension) {
		fatal("Mismatched vector size, z: %zu a: %zu", z->dimension, a->dimension);
	}
#endif
	memcpy(a->data, z->data, z->dimension*sizeof(float));
}
void nn_none_fn_d(Vector* z, Vector* d) {
#ifndef NO_BOUND_CHECK
	if (z->dimension != d->dimension) {
		fatal("Mismatched vector size, z: %zu d: %zu", z->dimension, d->dimension);
	}
#endif
	for (size_t i = 0; i < z->dimension; i++) {
		d->data[i] = 1.0f;
	}
}

//////////
// ReLU //
//////////

void nn_relu(Vector* z, Vector* a) {
#ifndef NO_BOUND_CHECK
	if (z->dimension != a->dimension) {
		fatal("Mismatched vector size, z: %zu a: %zu", z->dimension, a->dimension);
	}
#endif
#ifdef SIMD_AVX
	AVX256 vlbound = AVX256ZERO;
	for (size_t i = 0; i < z->dimension; i+=8) {
		avxmm256_unload_ptr(
				avxmm256_max(avxmm256_load_ptr(&(z->data[i])), vlbound), &(a->data[i])
				);
	}
#else
	for (size_t i = 0; i < z->dimension; i++) {
		a->data[i] = fmaxf(z->data[i], 0.0f);
	}
#endif
}
void nn_relu_d(Vector* z, Vector* d) {
#ifndef NO_BOUND_CHECK
	if (z->dimension != d->dimension) {
		fatal("Mismatched vector size, z: %zu d: %zu", z->dimension, d->dimension);
	}
#endif
#ifdef SIMD_AVX
	AVX256 vlbound = AVX256ZERO;
	AVX256 vmask = AVX256ONE;
	for (size_t i = 0; i < z->dimension; i+=8) {
		avxmm256_unload_ptr(
				avxmm256_mask(
					avxmm256_cmp_GT(avxmm256_load_ptr(&(z->data[i])), vlbound),
					vmask
					)
					, &(d->data[i])
				);
	}
#else
	for (size_t i = 0; i < z->dimension; i++) {
		d->data[i] = z->data[i] > 0.0f;
	}
#endif
}
void nn_crelu(Vector* z, Vector* a) {
#ifndef NO_BOUND_CHECK
	if (z->dimension != a->dimension) {
		fatal("Mismatched vector size, z: %zu a: %zu", z->dimension, a->dimension);
	}
#endif
	for (size_t i = 0; i < z->dimension; i++) {
		if (z->data[i] > 0) a->data[i] = fminf(z->data[i], 1.0f);
	}
}
void nn_crelu_d(Vector* z, Vector* d) {
#ifndef NO_BOUND_CHECK
	if (z->dimension != d->dimension) {
		fatal("Mismatched vector size, z: %zu d: %zu", z->dimension, d->dimension);
	}
#endif
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
#ifndef NO_BOUND_CHECK
	if (z->dimension != a->dimension) {
		fatal("Mismatched vector size, z: %zu a: %zu", z->dimension, a->dimension);
	}
#endif
	for (size_t i = 0; i < z->dimension; i++) {
		a->data[i] = _sigmoid(z->data[i]);
	}
}
void nn_sigmoid_d(Vector* z, Vector* d) {
#ifndef NO_BOUND_CHECK
	if (z->dimension != d->dimension) {
		fatal("Mismatched vector size, z: %zu d: %zu", z->dimension, d->dimension);
	}
#endif
	for (size_t i = 0; i < z->dimension; i++) {
		float sig = _sigmoid(z->data[i]);
		d->data[i] = sig*(1-sig);
	}
}

/////////////
// Softmax //
/////////////

void nn_softmax(Vector* z, Vector* a) {
	// Find largest element of the vector
	float z_max = 0;
	for (size_t i = 0; i < z->dimension; i++) {
		if (z->data[i] > z_max) {
			z_max = z->data[i];
		}
		debug("z[i]: %.10f", z->data[i]);
	}

	float exp_sum = 0;
	for (size_t i = 0; i < z->dimension; i++) {
		float z_exp = exp(z->data[i] - z_max);
		exp_sum += z_exp;
		a->data[i] = z_exp;
	}
	debug("exp_sum: %.10f", exp_sum);
	for (size_t i = 0; i < z->dimension; i++) {
		debug("a[i]: %.10f", a->data[i]);
		a->data[i] /= exp_sum;
	}
}

/////////////
// Logging //
/////////////

void nn_logging_fn(Vector* z, Vector* a) {
	info("Layer Info:");
	for (size_t i = 0; i < z->dimension; i++) {
		printr("node[%zu]: %f\n", i, z->data[i]);
		a->data[i] = z->data[i];
	}
}
