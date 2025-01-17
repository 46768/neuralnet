#include "ffn_bpropagate.h"

#include <stdint.h>

#include "ffn_fpropagate.h"

#include "logger.h"

void _ffn_next_error(FFN* nn, FFNMempool* pool, int nxt_idx) {
	debug("calcing for l %d", nxt_idx);
	debug("err_coef[%d]:", nxt_idx);
	for (size_t y = 0; y < pool->err_coef[nxt_idx]->sy; y++) {
		for (size_t x = 0; x < pool->err_coef[nxt_idx]->sx; x++) {
			printr_d("%f ", matrix_get(pool->err_coef[nxt_idx], x, y));
		}
		newline_d();
	}
	// err_(l-1) = ((da[l-1]/dz[l-1]) hdm_p w[l-1]^T) * err_l
	matrix_transpose_ip(nn->weights[nxt_idx], pool->weight_trsp[nxt_idx]); // w[l-1]^T
	debug("weight_trsp[%d]:", nxt_idx);
	for (size_t y = 0; y < pool->weight_trsp[nxt_idx]->sy; y++) {
		for (size_t x = 0; x < pool->weight_trsp[nxt_idx]->sx; x++) {
			printr_d("%f ", matrix_get(pool->weight_trsp[nxt_idx], x, y));
		}
		newline_d();
	}
	nn->layer_activation_d[nxt_idx](pool->preactivations[nxt_idx], pool->a_deriv[nxt_idx]);
	//^ da[l-1]/dz[l-1]
	debug("a_deriv[%d]:", nxt_idx);
	for (size_t i = 0; i < pool->a_deriv[nxt_idx]->dimension; i++) {
		printr_d("%f ", pool->a_deriv[nxt_idx]->data[i]);
	}
	newline_d();
	vec_matrix_hadamard_ip(
			pool->a_deriv[nxt_idx],
			pool->weight_trsp[nxt_idx],
			pool->err_coef[nxt_idx]
			); // a_deriv hdm_p w_t
	debug("err_coef[%d]:", nxt_idx);
	for (size_t y = 0; y < pool->err_coef[nxt_idx]->sy; y++) {
		for (size_t x = 0; x < pool->err_coef[nxt_idx]->sx; x++) {
			printr_d("%f ", matrix_get(pool->err_coef[nxt_idx], x, y));
		}
		newline_d();
	}
	matrix_vec_mul_ip(
			pool->err_coef[nxt_idx],
			pool->gradient_b[nxt_idx+1],
			pool->gradient_b[nxt_idx]
			); // coef * err_l
}

float ffn_bpropagate(
		FFN* nn,
		FFNMempool* pool,
		Vector* input,
		Vector* target,
		float learning_rate
		) {
	if (!nn->immutable) {
		error("Network is mutable");
		return 1000000.0f;
	}
	size_t L = nn->hidden_size;

	// Forward propagation
	ffn_fpropagate(nn, pool, input);

	// Back propagation variables
	Vector* a_driv_L = pool->a_deriv_L;
	Vector* gradient_aL_C = pool->gradient_aL_C;
	nn->cost_fn_d(pool->activations[L-1], target, gradient_aL_C);
	nn->layer_activation_d[L-2](
			pool->preactivations[L-1],
			a_driv_L);
	debug("a_driv_L:");
	for (size_t i = 0; i < a_driv_L->dimension; i++) {
		printr_d("%f ", a_driv_L->data[i]);
	}
	newline_d();
	debug("gradiant_aL_C:");
	for (size_t i = 0; i < gradient_aL_C->dimension; i++) {
		printr_d("%f ", gradient_aL_C->data[i]);
	}
	newline_d();
	vec_mul_ip(a_driv_L, gradient_aL_C, pool->gradient_b[L-1]);
	float loss = nn->cost_fn(target, pool->activations[L-1]);
	//info("Network loss: %f", loss);

	// Backward propagation
	// Going from L-1 to 1
	for (size_t l = L-2; l != SIZE_MAX; l--) {
		// Get previous layer error signal
		Vector* ld_l1 = pool->gradient_b[l+1];
		debug("ld_l1:");
		for (size_t i = 0; i < ld_l1->dimension; i++) {
			printr_d("%f ", ld_l1->data[i]);
		}
		newline_d();
		// Calculate weight gradient
		// Storing gradient as to not skew next layer's error
		column_row_vec_mul_ip(ld_l1, pool->activations[l], pool->gradient_w[l]);
		// Storing the error signal for bias update
		_ffn_next_error(nn, pool, l);
	}

	// Apply backward propagation gradient
	// Going from L-1 to 0
	for (size_t l = L-2; l != SIZE_MAX; l--) {
		Matrix* gradient_w_l = pool->gradient_w[l];
		Vector* gradient_b_l = pool->gradient_b[l];

		debug("Weight gradient of l %d:", l);
		for (size_t y = 0; y < gradient_w_l->sy; y++) {
			for (size_t x = 0; x < gradient_w_l->sx; x++) {
				printr_d("%f ", matrix_get(gradient_w_l, x, y));
			}
			newline_d();
		}
		debug("Bias gradient of l %d:", l);
		for (size_t i = 0; i < gradient_b_l->dimension; i++) {
			printr_d("%f ", gradient_b_l->data[i]);
		}
		newline_d();

		// Apply weight gradient
		debug("Weight gradient applied to l %d:", l);
		for (size_t x = 0; x < gradient_w_l->sx; x++) {
			(nn->biases)[l]->data[x] -= learning_rate*gradient_b_l->data[x];
			for (size_t y = 0; y < gradient_w_l->sy; y++) {
				printr_d("%f ", -learning_rate*matrix_get(gradient_w_l, x, y));
				(nn->weights)[l]->data[y*gradient_w_l->sx + x] -= 
					learning_rate*matrix_get(gradient_w_l, x, y);
			}
			newline_d();
		}
		newline_d();
		debug("Bias gradient applied to l %d:", l);
		for (size_t x = 0; x < gradient_w_l->sx; x++) {
			printr_d("%f\n", -learning_rate*gradient_b_l->data[x]);
		}
		newline_d();
	}

	return loss;
}
