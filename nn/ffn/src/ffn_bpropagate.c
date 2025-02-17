#include "ffn_bpropagate.h"

#include "matrix.h"

void _ffn_next_error(
		FFNParameterPool* papool,
		FFNPropagationPool* prpool,
		FFNGradientPool* gpool,
		FFNIntermediatePool* ipool,
		int nxt_idx) {
	// err_(l-1) = ((da[l-1]/dz[l-1]) hdm_p w[l-1]^T) * err_l
	matrix_transpose_ip(&(papool->weights[nxt_idx]), &(ipool->weight_trsp[nxt_idx])); // w[l-1]^T
	papool->activation_fn_d[nxt_idx](&(prpool->preactivations[nxt_idx]), &(ipool->a_deriv[nxt_idx]));
	//^ da[l-1]/dz[l-1]
	vec_matrix_hadamard_ip(
			&(ipool->a_deriv[nxt_idx]),
			&(ipool->weight_trsp[nxt_idx]),
			&(ipool->err_coef[nxt_idx])
			); // a_deriv hdm_p w_t
	matrix_vec_mul_ip(
			&(ipool->err_coef[nxt_idx]),
			&(gpool->gradient_b[nxt_idx+1]),
			&(gpool->gradient_b[nxt_idx])
			); // coef * err_l
}

void ffn_apply_gradient(FFNParams* init_data, FFNParameterPool* papool, FFNGradientPool* gpool, float learning_rate) {
	// Apply backward propagation gradient
	// Going from L-1 to 0
	size_t L = papool->layer_cnt;
	for (int l = 0; l < ((int)L-1); l++) {
		if (init_data->hidden_layers[l+1]->l_type == PassThrough) {
			continue;
		}
		Matrix* gradient_w_l = &(gpool->gradient_w[l]);
		Matrix* weight_l = &(papool->weights[l]);
		Vector* gradient_b_l = &(gpool->gradient_b[l+1]);
		Vector* bias_l = &(papool->biases[l]);

		// Apply weight gradient
		matrix_coef_add_ip(gradient_w_l, weight_l, -learning_rate, weight_l);
		vec_coef_add_ip(gradient_b_l, bias_l, -learning_rate, bias_l);

		/*
		for (size_t y = 0; y < gradient_w_l->sy; y++) {
			(nn->biases)[l]->data[y] -= learning_rate*gradient_b_l->data[y];
			for (size_t x = 0; x < gradient_w_l->sx; x++) {
				*matrix_get_ptr(nn->weights[l], x, y) -= 
					learning_rate*matrix_get(gradient_w_l, x, y);
			}
		}
		*/
	}
}

float ffn_get_param_change(
		FFNParams* init_data,
		FFNParameterPool* papool,
		FFNPropagationPool* prpool,
		FFNGradientPool* gpool,
		FFNIntermediatePool* ipool,
		Vector* target
		) {
	size_t L = init_data->hidden_size;

	// Back propagation variables
	Vector* a_driv_L = &(ipool->a_deriv[L-1]);
	Vector* gradient_aL_C = &(ipool->a_deriv[L]);

	papool->cost_fn_d(&(prpool->activations[L-1]), target, gradient_aL_C);
	papool->activation_fn_d[L-2](
			&(prpool->preactivations[L-1]),
			a_driv_L);
	vec_mul_ip(a_driv_L, gradient_aL_C, &(gpool->gradient_b[L-1]));
	float loss = papool->cost_fn(&(prpool->activations[L-1]), target);
	//info("Network loss: %f", loss);

	// Backward propagation
	// Going from L-1 to 1
	for (int l = L-2; l >= 0; l--) {
		// Get previous layer error signal
		Vector* ld_l1 = &(gpool->gradient_b[l+1]);
		// Calculate weight gradient
		column_row_vec_mul_ip(ld_l1, &(prpool->activations[l]), &(gpool->gradient_w[l]));
		// Storing the error signal for bias gradient
		_ffn_next_error(papool, prpool, gpool, ipool, l);
	}

	return loss;
}
