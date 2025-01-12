#include "ffn_bpropagate.h"

#include "logger.h"
#include "allocator.h"

float _ffn_cost_mse(Vector* actual, Vector* target) {
	Vector* diff = vec_sub(actual, target);
	float cost = vec_dot(diff, diff)/diff->dimension;
	vec_deallocate(diff);
	return cost;
}

void _ffn_bfpropagate(FFN* nn, Vector* input, Vector*** a, Vector*** z) {
	Matrix** weights = nn->weights;
	Vector** biases = nn->biases;

	// Dereference activation and preactivation pointers
	Vector** activations = *a;
	Vector** preactivations = *z;

	// Check input compatibility
	if (input->dimension != nn->hidden_layers[0]) {
		fatal("Input vector size mismatched with input node count, %d to %d", input->dimension,
				nn->hidden_layers[0]);
		exit(1);
	}

	// Set activations and preactivations
	activations[0] = vec_dup(input);
	preactivations[0] = vec_dup(input);

	// Propagate input to each layer
	for (int l = 0; l < nn->hidden_size-1; l++) {
		Matrix* weight = weights[l];
		Vector* bias = biases[l];

		Vector* activation = vec_zero(bias->dimension);
		Vector* preactivation = vec_zero(bias->dimension);

		// Matrix vector multiplication
		for (int y = 0; y < weight->sy; y++) {
			float a_j = (bias->data)[y];
			for (int x = 0; x < weight->sx; x++) {
				a_j += (activations[l]->data)[x]*matrix_get(weight, x, y);
			}
			(activation->data)[y] = a_j > 0 ? a_j : 0;
			(preactivation->data)[y] = a_j;
		}

		// Assign next layer activation
		preactivations[l+1] = preactivation;
		activations[l+1] = activation;
	}
}

Vector* _ffn_relu_derivative(Vector* wrt) {
	// dReLU(x)/dx = x > 0 ? 1 : 0
	Vector* driv = vec_zero(wrt->dimension);

	for (int i = 0; i < wrt->dimension; i++) {
		if (wrt->data[i] > 0) {
			driv->data[i] = 1;
		}
	}

	return driv;
}

Vector* _ffn_mse_derivative(Vector* target, Vector* wrt) {
	// dMse/dwrt_x = 2(wrt_x - target_x)/wrt.dim
	Vector* driv = vec_zero(target->dimension);
	float driv_coef = 2/(float)target->dimension;

	for (int i = 0; i < target->dimension; i++) {
		driv->data[i] = driv_coef*(wrt->data[i] - target->data[i]);
	}

	return driv;
}

Vector* _ffn_next_error(FFN* nn, Vector** z, Vector* cur_error, int nxt_idx) {
	// err_(l-1) = ((da[l-1]/dz[l-1]) hdm_p w[l-1]^T) * err_l
	Matrix* weight_trsp = matrix_transpose(nn->weights[nxt_idx]); // w[l-1]^T
	Vector* relu_deriv = _ffn_relu_derivative(z[nxt_idx]); // da[l-1]/dz[l-1]
	Matrix* err_coef = vec_matrix_hadamard(relu_deriv, weight_trsp); // drelu hdm_p w_t)
	Vector* nxt_err = matrix_vec_mul(err_coef, cur_error); // coef * err_l

	// Pointer cleanup
	matrix_deallocate(weight_trsp);
	matrix_deallocate(err_coef);
	vec_deallocate(relu_deriv);

	return nxt_err;
}

void ffn_bpropagate(FFN* nn, Vector* input, Vector* target, float learning_rate) {
	if (!nn->immutable) {
		error("Network is mutable");
		return;
	}
	size_t L = nn->hidden_size;

	// Forward propagation
	Vector** preactivation = (Vector**)calloc(L, sizeof(Vector*));
	Vector** activation = (Vector**)calloc(L, sizeof(Vector*));
	_ffn_bfpropagate(nn, input, &activation, &preactivation);

	// Back propagation variables
	Vector* gradient_aL_C = _ffn_mse_derivative(target, activation[L-1]);
	Vector* relu_driv_L = _ffn_relu_derivative(preactivation[L-1]);
	Vector* ld_l = vec_mul(relu_driv_L, gradient_aL_C);
	vec_deallocate(relu_driv_L);
	vec_deallocate(gradient_aL_C);
	
	Matrix** gradient_w = (Matrix**)calloc(L, sizeof(Matrix*));
	float loss = _ffn_cost_mse(target, activation[L-1]);

	// Backward propagation
	// Going from L-1 to 0
	for (int l = L-2; l >= 0; l--) {
		// Calculate weight gradient
		// Storing gradient as to not skew next layer's error
		gradient_w[l] = column_row_vec_mul(ld_l, activation[l]);

		Vector* ld_temp = ld_l;
		ld_l = _ffn_next_error(nn, preactivation, ld_l, l);
		vec_deallocate(ld_temp);
	}

	// Apply backward propagation gradient
	// Going from L-1 to 0
	for (int l = L-2; l >= 0; l--) {
		Matrix* gradient_w_l = gradient_w[l];

		// Apply weight gradient
		for (int x = 0; x < gradient_w_l->sx; x++) {
			for (int y = 0; y < gradient_w_l->sy; y++) {
				(nn->weights)[l]->data[y*gradient_w_l->sx + x] -= 
					learning_rate*matrix_get(gradient_w_l, x, y);
			}
		}

		// Deallocate weight gradient
		matrix_deallocate(gradient_w[l]);
	}

	// Cleanup
	for (int l = 0; l < L; l++) {
		vec_deallocate(preactivation[l]);
		vec_deallocate(activation[l]);
	}
	deallocate(preactivation);
	deallocate(activation);
	deallocate(gradient_w);
	vec_deallocate(ld_l);
}
