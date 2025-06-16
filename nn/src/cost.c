#include "cost.h"

#include <math.h>

float cost_mse(Vector* vin, Vector *target) {
	float cost = 0;

	get_vec_ctx(vi, vin);
	get_vec_ctx(tg, target);

	for (uint32_t i = 0; i < vi.size; i++) {
		float diff = vec_idx(vi, i) - vec_idx(tg, i);
		cost += diff*diff;
	}

	return cost/(float)(vi.size);
}

void cost_mse_d(Vector* vin, Vector* target, Vector *vout) {
	get_vec_ctx(vi, vin);
	get_vec_ctx(tg, target);
	get_vec_ctx(vo, vout);

	float deriv_coef = 2/(float)(vi.size);

	for (uint32_t i = 0; i < vi.size; i++) {
		vec_idx(vo, i) = deriv_coef*(vec_idx(vi, i) - vec_idx(tg, i));
	}
}


float cost_ccel(Vector* vin, Vector* target) {
	get_vec_ctx(vi, vin);
	get_vec_ctx(tg, target);

	float cost = 0;

	float vin_max = -2147483647.0f;
	for (uint32_t i = 0; i < vi.size; i++) {
		if (vec_idx(vi, i) > vin_max) {
			vin_max = vec_idx(vi, i);
		}
	}

	float exp_sum = 0;
	for (uint32_t i = 0; i < vi.size; i++) {
		exp_sum += exp(vec_idx(vi, i) - vin_max);
	}
	
	float ccel_const = vin_max + logf(exp_sum);

	for (uint32_t i = 0; i < vi.size; i++) {
		cost += vec_idx(tg, i) * (vec_idx(vi, i) - ccel_const);
	}

	return -cost;
}

void cost_ccel_d(Vector* vin, Vector* target, Vector* vout) {
	get_vec_ctx(vi, vin);
	get_vec_ctx(tg, target);
	get_vec_ctx(vo, vout);

	for (uint32_t i = 0; i < vi.size; i++) {
		vec_idx(vo, i) = vec_idx(tg, i) - vec_idx(vi, i);
	}
}
