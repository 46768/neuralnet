#include "ffn_bpropagate.h"

#include "logger.h"
#include "allocator.h"

void _ffn_bfpropagate(FFN* nn, Vector* input, Vector*** a, Vector*** z) {
	Matrix** weights = nn->weights;
	Vector** biases = nn->biases;

	// Dereference activation and preactivation pointers
	Vector** activations = *a;
	Vector** preactivations = *z;

	// Check input compatibility
	if (input->dimension != (nn->hidden_layers[0])->node_cnt) {
		fatal("Input vector size mismatched with input node count, %d to %d", input->dimension,
				(nn->hidden_layers[0])->node_cnt);
		exit(1);
	}

	// Set activations and preactivations
	activations[0] = vec_dup(input);
	preactivations[0] = vec_dup(input);

	// Propagate input to each layer
	for (int l = 0; l < nn->hidden_size-1; l++) {
		Matrix* weight = weights[l];
		Vector* bias = biases[l];

		Vector* preactivation = vec_zero(bias->dimension);

		// Matrix vector multiplication
		for (int y = 0; y < weight->sy; y++) {
			float a_j = (bias->data)[y];
			for (int x = 0; x < weight->sx; x++) {
				a_j += (activations[l]->data)[x]*matrix_get(weight, x, y);
			}
			(preactivation->data)[y] = a_j;
		}
		Vector* activation = nn->layer_activation[l](preactivation);

		// Assign next layer activation
		preactivations[l+1] = preactivation;
		activations[l+1] = activation;
	}
}

Vector* _ffn_next_error(FFN* nn, Vector** z, Vector* cur_error, int nxt_idx) {
	// err_(l-1) = ((da[l-1]/dz[l-1]) hdm_p w[l-1]^T) * err_l
	Matrix* weight_trsp = matrix_transpose(nn->weights[nxt_idx]); // w[l-1]^T
	Vector* a_deriv = nn->layer_activation_d[nxt_idx](z[nxt_idx]); // da[l-1]/dz[l-1]
	Matrix* err_coef = vec_matrix_hadamard(a_deriv, weight_trsp); // a_deriv hdm_p w_t
	Vector* nxt_err = matrix_vec_mul(err_coef, cur_error); // coef * err_l

	// Pointer cleanup
	matrix_deallocate(weight_trsp);
	matrix_deallocate(err_coef);
	vec_deallocate(a_deriv);

	return nxt_err;
}

float ffn_bpropagate(FFN* nn, Vector* input, Vector* target, float learning_rate) {
	if (!nn->immutable) {
		error("Network is mutable");
		return -1.0f;
	}
	size_t L = nn->hidden_size;

	// Forward propagation
	Vector** preactivation = (Vector**)calloc(L, sizeof(Vector*));
	Vector** activation = (Vector**)calloc(L, sizeof(Vector*));
	_ffn_bfpropagate(nn, input, &activation, &preactivation);

	// Back propagation variables
	Vector* gradient_aL_C = nn->cost_fn_d(activation[L-1], target);
	Vector* a_driv_L = nn->layer_activation_d[L-2](preactivation[L-1]);
	
	Matrix** gradient_w = (Matrix**)callocate(L, sizeof(Matrix*));
	Vector** gradient_b = (Vector**)callocate(L, sizeof(Vector*));
	gradient_b[L-1] = vec_mul(a_driv_L, gradient_aL_C);
	debug("a_driv_L:");
	for (int i = 0; i < a_driv_L->dimension; i++) {
		printr_d("%f ", a_driv_L->data[i]);
	}
	newline_d();
	debug("gradiant_aL_C:");
	for (int i = 0; i < gradient_aL_C->dimension; i++) {
		printr_d("%f ", gradient_aL_C->data[i]);
	}
	newline_d();
	vec_deallocate(a_driv_L);
	vec_deallocate(gradient_aL_C);
	float loss = nn->cost_fn(target, activation[L-1]);
	//info("Network loss: %f", loss);

	// Backward propagation
	// Going from L-1 to 1
	for (int l = L-2; l >= 0; l--) {
		// Get previous layer error signal
		Vector* ld_l1 = gradient_b[l+1];
		debug("ld_l1:");
		for (int i = 0; i < ld_l1->dimension; i++) {
			printr_d("%f ", ld_l1->data[i]);
		}
		newline_d();
		// Calculate weight gradient
		// Storing gradient as to not skew next layer's error
		gradient_w[l] = column_row_vec_mul(ld_l1, activation[l]);
		// Storing the error signal for bias update
		gradient_b[l] = _ffn_next_error(nn, preactivation, ld_l1, l);
	}

	// Apply backward propagation gradient
	// Going from L-1 to 0
	for (int l = L-2; l >= 0; l--) {
		Matrix* gradient_w_l = gradient_w[l];
		Vector* gradient_b_l = gradient_b[l];

		// Apply weight gradient
		debug("Weight gradient applied to l %d:", l);
		for (int x = 0; x < gradient_w_l->sx; x++) {
			(nn->biases)[l]->data[x] -= learning_rate*gradient_b_l->data[x];
			for (int y = 0; y < gradient_w_l->sy; y++) {
				printr_d("%f ", -learning_rate*matrix_get(gradient_w_l, x, y));
				(nn->weights)[l]->data[y*gradient_w_l->sx + x] -= 
					learning_rate*matrix_get(gradient_w_l, x, y);
			}
			newline_d();
		}
		newline_d();
		debug("Bias gradient applied to l %d:", l);
		for (int x = 0; x < gradient_w_l->sx; x++) {
			printr_d("%f\n", -learning_rate*gradient_b_l->data[x]);
		}
		newline_d();

		// Deallocate gradients
		matrix_deallocate(gradient_w[l]);
		vec_deallocate(gradient_b[l]);
	}

	// Cleanup
	for (int l = 0; l < L; l++) {
		vec_deallocate(preactivation[l]);
		vec_deallocate(activation[l]);
	}
	vec_deallocate(gradient_b[L-1]);
	deallocate(preactivation);
	deallocate(activation);
	deallocate(gradient_w);
	deallocate(gradient_b);

	return loss;
}
