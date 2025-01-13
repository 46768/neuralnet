#include "cost.h"

#include <math.h>

#include "activation.h"
#include "logger.h"

CostFn resolve_cost_fn(CostFnEnum fn_type) {
	switch (fn_type) {
		case MSE:
			return nn_mse;
		case CEL:
			return nn_cel;
		default:
			return NULL;
	}
}

CostFnD resolve_cost_fn_d(CostFnEnum fn_type) {
	switch (fn_type) {
		case MSE:
			return nn_mse_d;
		case CEL:
			return nn_cel_d;
		default:
			return NULL;
	}
}

////////////////////////
// Mean Squared Error //
////////////////////////

float nn_mse(Vector* actual, Vector* target) {
	Vector* diff = vec_sub(actual, target);
	float cost = vec_dot(diff, diff)/diff->dimension;
	vec_deallocate(diff);
	return cost;
}
Vector* nn_mse_d(Vector* actual, Vector* target) {
	// dMse/dwrt_x = 2(wrt_x - target_x)/wrt.dim
	Vector* driv = vec_zero(target->dimension);
	float driv_coef = 2/(float)target->dimension;

	for (int i = 0; i < target->dimension; i++) {
		debug("deriv[%d]: %f*(%f - %f)", i, driv_coef, actual->data[i], target->data[i]);
		driv->data[i] = driv_coef*(actual->data[i] - target->data[i]);
	}

	return driv;
}

////////////////////////
// Cross Entropy Loss //
////////////////////////

float nn_cel(Vector* actual, Vector* target) {
	float loss = 0;
	Vector* softmax_a = nn_softmax(actual);
	for (int i = 0; i < actual->dimension; i++) {
		loss -= target->data[i] * log(softmax_a->data[i]);
	}

	vec_deallocate(softmax_a);
	return loss;
}
Vector* nn_cel_d(Vector* actual, Vector* target) {
	Vector* driv = vec_zero(actual->dimension);
	Vector* softmax_a = nn_softmax(actual);

	for (int i = 0; i < actual->dimension; i++) {
		driv->data[i] = softmax_a->data[i] - target->data[i];
	}

	vec_deallocate(softmax_a);
	return driv;
}
