#include "cost.h"

#include <math.h>

#include "logger.h"

CostFn resolve_cost_fn(CostFnEnum fn_type) {
	switch (fn_type) {
		case MSE:
			return nn_mse;
		case CCE:
			return nn_ccel;
		case BCE:
			return nn_bcel;
		default:
			return NULL;
	}
}

CostFnD resolve_cost_fn_d(CostFnEnum fn_type) {
	switch (fn_type) {
		case MSE:
			return nn_mse_d;
		case CCE:
			return nn_ccel_d;
		case BCE:
			return nn_bcel_d;
		default:
			return NULL;
	}
}

////////////////////////
// Mean Squared Error //
////////////////////////

float nn_mse(Vector* actual, Vector* target) {
	if (actual->dimension != target->dimension) {
		fatal("Mismatched z to t size: z: %zu t: %zu",
				actual->dimension,
				target->dimension
			 );
	}

	float cost = 0;
	for (size_t i = 0; i < actual->dimension; i++) {
		float diff = actual->data[i] - target->data[i];
		cost += diff*diff;
		debug("accumulated cost: %f", cost);
	}
	return cost/(float)actual->dimension;
}
void nn_mse_d(Vector* actual, Vector* target, Vector* driv) {
	if (actual->dimension != target->dimension) {
		fatal("Mismatched z to t size: z: %zu t: %zu",
				actual->dimension,
				target->dimension
			 );
	}
	if (actual->dimension != driv->dimension) {
		fatal("Mismatched z to d size: z: %zu d: %zu",
				actual->dimension,
				driv->dimension
			 );
	}

	// dMse/dwrt_x = 2(wrt_x - target_x)/wrt.dim
	float driv_coef = 2/(float)target->dimension;

	for (size_t i = 0; i < target->dimension; i++) {
		//debug("deriv[%d]: %f*(%f - %f)", i, driv_coef, actual->data[i], target->data[i]);
		driv->data[i] = driv_coef*(actual->data[i] - target->data[i]);
	}
}

////////////////////////////////////
// Categorical Cross Entropy Loss //
////////////////////////////////////

float nn_ccel(Vector* actual, Vector* target) {
	if (actual->dimension != target->dimension) {
		fatal("Mismatched z to t size, z: %zu t: %zu",
				actual->dimension,
				target->dimension
			 );
	}

	float loss = 0;
	float exp_sum = 0;
	float z_max = -INFINITY;
	for (size_t i = 0; i < actual->dimension; i++) {
		if (actual->data[i] > z_max) {
			z_max = actual->data[i];
		}
	}
	for (size_t i = 0; i < actual->dimension; i++) {
		exp_sum += exp(actual->data[i] - z_max);
	}
	float cel_offset = log(exp_sum);
	for (size_t i = 0; i < actual->dimension; i++) {
		debug("loss_i: %f", target->data[i] * -(actual->data[i] - z_max - cel_offset));
		loss -= target->data[i] * (actual->data[i] - z_max - cel_offset);
	}

	return loss;
}
void nn_ccel_d(Vector* actual, Vector* target, Vector* res) {
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
	float exp_sum = 0;
	float z_max = -INFINITY;
	for (size_t i = 0; i < actual->dimension; i++) {
		if (actual->data[i] > z_max) {
			z_max = actual->data[i];
		}
	}
	for (size_t i = 0; i < actual->dimension; i++) {
		exp_sum += exp(actual->data[i] - z_max);
	}

	for (size_t i = 0; i < actual->dimension; i++) {
		float pow_i = actual->data[i] - z_max;
		float exp_i = exp(pow_i);
		res->data[i] = -(exp_sum / exp_i) * target->data[i] * pow_i * exp_i;
	}
}

///////////////////////////////
// Binary Cross Entropy Loss //
///////////////////////////////

float nn_bcel(Vector* actual, Vector* target) {
	if (actual->dimension != target->dimension) {
		fatal("Mismatched z to t size, z: %zu t: %zu",
				actual->dimension,
				target->dimension
			 );
	}
	if (actual->dimension != 1) {
		fatal("Using BCEL for non 1D vector, use CCEL instead");
	}
	float clipped_a = fmax(fmin(actual->data[0], 1.0-1e-15), 1e-15);
	debug("clipped activation: %.10f", clipped_a);
	debug("target: %.10f", target->data[0]);
	//float clipped_a = actual->data[0];
	if (target->data[0] == 0.0f) {
		return -log(1 - clipped_a);
	} else {
		return -log(clipped_a);
	}
}
void nn_bcel_d(Vector* actual, Vector* target, Vector* res) {
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
	if (actual->dimension != 1) {
		fatal("Using BCELd for non 1D vector, use CCELd instead");
	}
	float clipped_a = fmax(fmin(actual->data[0], 1.0-1e-15), 1e-15);
	if (target->data[0] == 0) {
		res->data[0] = -1/(1 - clipped_a);
	} else {
		res->data[0] = -1/clipped_a;
	}
}
