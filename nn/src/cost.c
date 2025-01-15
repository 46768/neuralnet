#include "cost.h"

#include <math.h>

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
	if (actual->dimension != target->dimension) {
		fatal("Mismatched a to t size: a: %zu t: %zu",
				actual->dimension,
				target->dimension
			 );
	}
	float cost = 0;
	for (int i = 0; i < actual->dimension; i++) {
		cost += (actual->data[i]-target->data[i])/(float)actual->dimension;
	}
	return cost;
}
void nn_mse_d(Vector* actual, Vector* target, Vector* driv) {
	// dMse/dwrt_x = 2(wrt_x - target_x)/wrt.dim
	float driv_coef = 2/(float)target->dimension;

	for (int i = 0; i < target->dimension; i++) {
		//debug("deriv[%d]: %f*(%f - %f)", i, driv_coef, actual->data[i], target->data[i]);
		driv->data[i] = driv_coef*(actual->data[i] - target->data[i]);
	}
}

////////////////////////
// Cross Entropy Loss //
////////////////////////

float nn_cel(Vector* actual, Vector* target) {
	if (actual->dimension != target->dimension) {
		fatal("Mismatched z to t size, z: %zu t: %zu",
				actual->dimension,
				target->dimension
			 );
	}
	if (actual->dimension == 1) {
		return log(1+exp(-actual->data[0])) - target->data[0];
	}
	float loss = 0;
	float exp_sum = 0;
	float z_max = -INFINITY;
	for (int i = 0; i < actual->dimension; i++) {
		if (actual->data[i] > z_max) {
			z_max = actual->data[i];
		}
	}
	for (int i = 0; i < actual->dimension; i++) {
		exp_sum += exp(actual->data[i] - z_max);
	}
	float cel_offset = log(exp_sum);
	for (int i = 0; i < actual->dimension; i++) {
		loss -= target->data[i] * (actual->data[i] - z_max - cel_offset);
	}

	return loss;
}
void nn_cel_d(Vector* actual, Vector* target, Vector* res) {
	if (actual->dimension != target->dimension) {
		fatal("Mismatched z to t size, z: %zu t: %zu",
				actual->dimension,
				target->dimension
			 );
	}
	if (actual->dimension != res->dimension) {
		fatal("Mismatched z to res size, z: %zu res: %zu",
				actual->dimension,
				res->dimension
			 );
	}
	if (actual->dimension == 1) {
		res->data[0] = ((float)1/(1+exp(-actual->data[0]))) - target->data[0];
		return;
	}
	float exp_sum = 0;
	float z_max = -INFINITY;
	for (int i = 0; i < actual->dimension; i++) {
		if (actual->data[i] > z_max) {
			z_max = actual->data[i];
		}
	}
	for (int i = 0; i < actual->dimension; i++) {
		exp_sum += exp(actual->data[i] - z_max);
	}

	for (int i = 0; i < actual->dimension; i++) {
		res->data[i] = (exp(actual->data[i] - z_max)/exp_sum) - target->data[i];
	}
}
