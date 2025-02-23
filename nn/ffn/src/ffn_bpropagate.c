#include "ffn_bpropagate.h"

#include "matrix.h"

void ffn_apply_gradient(FFNParams* init_data, FFNParameterPool* papool, FFNGradientPool* gpool, float learning_rate) {
	size_t L = papool->base.layer_cnt;
	for (int l = 0; l < ((int)L-1); l++) {
		if (init_data->hidden_layers[l+1]->l_type == PassThrough) {
			continue;
		}
		Matrix* gradient_w_l = &(gpool->gradient_w[l]);
		Vector* gradient_b_l = &(gpool->gradient_b[l+1]);

		Matrix* weight_l = &(papool->weights[l]);
		Vector* bias_l = &(papool->biases[l]);

		matrix_coef_add_ip(gradient_w_l, weight_l, -learning_rate, weight_l);
		vec_coef_add_ip(gradient_b_l, bias_l, -learning_rate, bias_l);
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
	Vector* a_deriv_l = &(ipool->a_deriv[L-1]);
	Vector* gradient_aL_C = a_deriv_l+1;
	Vector* model_output = &(prpool->activations[L-1]);
	Vector* bias_gradient = &(gpool->gradient_b[L-1]);
	Matrix* transpose = &(ipool->weight_trsp[L-2]);
	Matrix* error_coef = &(ipool->err_coef[L-2]);
	Vector* preactv = &(prpool->preactivations[L-2]);

	papool->cost_fn_d(model_output, target, gradient_aL_C);
	papool->activation_fn_d[L-2](
			model_output,
			a_deriv_l);

	vec_mul_ip(a_deriv_l, gradient_aL_C, bias_gradient);
	for (int l = L-2; l >= 0; l--) {
		column_row_vec_mul_ip(bias_gradient, &(prpool->activations[l]), &(gpool->gradient_w[l]));

		matrix_transpose_ip(&(papool->weights[l]), transpose);
		papool->activation_fn_d[l](preactv, a_deriv_l-1);
		vec_matrix_hadamard_ip(a_deriv_l-1, transpose, error_coef);
		matrix_vec_mul_ip(error_coef, bias_gradient, bias_gradient-1);

		// Step values to the previous layer
		bias_gradient--;
		a_deriv_l--;
		transpose--;
		error_coef--;
		preactv--;
	}

	return papool->cost_fn(model_output, target);
}
