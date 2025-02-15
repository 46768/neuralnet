#include "ffn_bpropagate.h"

#include "matrix.h"

#include "logger.h"

void _ffn_next_error(FFN* nn, FFNMempool* pool, int nxt_idx) {
	// err_(l-1) = ((da[l-1]/dz[l-1]) hdm_p w[l-1]^T) * err_l
	matrix_transpose_ip(nn->weights[nxt_idx], &(pool->intermediates->weight_trsp[nxt_idx])); // w[l-1]^T
	nn->layer_activation_d[nxt_idx](&(pool->propagation->preactivations[nxt_idx]), &(pool->intermediates->a_deriv[nxt_idx]));
	//^ da[l-1]/dz[l-1]
	vec_matrix_hadamard_ip(
			&(pool->intermediates->a_deriv[nxt_idx]),
			&(pool->intermediates->weight_trsp[nxt_idx]),
			&(pool->intermediates->err_coef[nxt_idx])
			); // a_deriv hdm_p w_t
	matrix_vec_mul_ip(
			&(pool->intermediates->err_coef[nxt_idx]),
			&(pool->gradients->gradient_b[nxt_idx+1]),
			&(pool->gradients->gradient_b[nxt_idx])
			); // coef * err_l
}

void ffn_apply_gradient(FFN* nn, FFNGradientPool* pool, float learning_rate) {
	// Apply backward propagation gradient
	// Going from L-1 to 0
	size_t L = nn->hidden_size;
	for (int l = (int)L-2; l >= 0; l--) {
		if (nn->hidden_layers[l+1]->l_type == PassThrough) {
			continue;
		}
		Matrix* gradient_w_l = &(pool->gradient_w[l]);
		Vector* gradient_b_l = &(pool->gradient_b[l+1]);

		// Apply weight gradient
		debug("Weight modification to layer[%d]", l);
		for (size_t y = 0; y < gradient_w_l->sy; y++) {
			(nn->biases)[l]->data[y] -= learning_rate*gradient_b_l->data[y];
			for (size_t x = 0; x < gradient_w_l->sx; x++) {
				*matrix_get_ptr(nn->weights[l], x, y) -= 
					learning_rate*matrix_get(gradient_w_l, x, y);
				printr_d("%f ", -learning_rate*matrix_get(gradient_w_l, x, y));
			}
			newline_d();
		}
		newline_d();
		debug("Bias modification to layer[%d]", l);
		for (size_t x = 0; x < gradient_w_l->sx; x++) {
			printr_d("%f ", -learning_rate*gradient_b_l->data[x]);
		}
		newline_d();
	}
}

float ffn_get_param_change(
		FFN* nn,
		FFNMempool* pool,
		Vector* target
		) {
	if (!nn->immutable) {
		error("Network is mutable");
		return 1000000.0f;
	}
	size_t L = nn->hidden_size;

	// Back propagation variables
	Vector* a_driv_L = &(pool->intermediates->a_deriv[L-1]);
	Vector* gradient_aL_C = &(pool->intermediates->a_deriv[L]);

	nn->cost_fn_d(&(pool->propagation->activations[L-1]), target, gradient_aL_C);
	nn->layer_activation_d[L-2](
			&(pool->propagation->preactivations[L-1]),
			a_driv_L);
	vec_mul_ip(a_driv_L, gradient_aL_C, &(pool->gradients->gradient_b[L-1]));
	float loss = nn->cost_fn(&(pool->propagation->activations[L-1]), target);
	//info("Network loss: %f", loss);

	// Backward propagation
	// Going from L-1 to 1
	for (int l = L-2; l >= 0; l--) {
		// Get previous layer error signal
		Vector* ld_l1 = &(pool->gradients->gradient_b[l+1]);
		// Calculate weight gradient
		column_row_vec_mul_ip(ld_l1, &(pool->propagation->activations[l]), &(pool->gradients->gradient_w[l]));
		// Storing the error signal for bias gradient
		_ffn_next_error(nn, pool, l);
	}

	return loss;
}
