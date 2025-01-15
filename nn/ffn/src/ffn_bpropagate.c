#include "ffn_bpropagate.h"

#include "ffn_fpropagate.h"

#include "logger.h"

void _ffn_next_error(FFN* nn, FFNMempool* pool, int nxt_idx) {
	// err_(l-1) = ((da[l-1]/dz[l-1]) hdm_p w[l-1]^T) * err_l
	matrix_transpose_ip(nn->weights[nxt_idx], pool->weight_trsp[nxt_idx]); // w[l-1]^T
	nn->layer_activation_d[nxt_idx](pool->preactivations[nxt_idx], pool->a_deriv[nxt_idx]);
	//^ da[l-1]/dz[l-1]
	vec_matrix_hadamard_ip(
			pool->a_deriv[nxt_idx],
			pool->weight_trsp[nxt_idx],
			pool->err_coef[nxt_idx]
			); // a_deriv hdm_p w_t
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
		return -1.0f;
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
	
	vec_mul_ip(a_driv_L, gradient_aL_C, pool->gradient_b[L-1]);
	float loss = nn->cost_fn(target, pool->activations[L-1]);
	//info("Network loss: %f", loss);

	// Backward propagation
	// Going from L-1 to 1
	for (int l = L-2; l >= 0; l--) {
		// Get previous layer error signal
		Vector* ld_l1 = pool->gradient_b[l+1];
		// Calculate weight gradient
		// Storing gradient as to not skew next layer's error
		column_row_vec_mul_ip(ld_l1, pool->activations[l], pool->gradient_w[l]);
		// Storing the error signal for bias update
		_ffn_next_error(nn, pool, l);
	}

	// Apply backward propagation gradient
	// Going from L-1 to 0
	for (int l = L-2; l >= 0; l--) {
		Matrix* gradient_w_l = pool->gradient_w[l];
		Vector* gradient_b_l = pool->gradient_b[l];

		// Apply weight gradient
		for (int x = 0; x < gradient_w_l->sx; x++) {
			(nn->biases)[l]->data[x] -= learning_rate*gradient_b_l->data[x];
			for (int y = 0; y < gradient_w_l->sy; y++) {
				(nn->weights)[l]->data[y*gradient_w_l->sx + x] -= 
					learning_rate*matrix_get(gradient_w_l, x, y);
			}
		}
	}

	return loss;
}
