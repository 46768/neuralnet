#include "ffn_util.h"

#include "logger.h"

#include "matrix.h"
#include "vector.h"

#include "ffn_mempool.h"

void ffn_dump_param(FFNParameterPool* papool) {
	info("Weights");
	for (int l = 0; l < ((int)(papool->base.layer_cnt)-1); l++) {
		info("Layer %d:", l);
		matrix_dump_raw(&(papool->weights[l]));
		newline();
	}
	newline();
	info("Biases");
	for (int l = 0; l < ((int)(papool->base.layer_cnt)-1); l++) {
		info("Layer %d:", l);
		vec_dump(&(papool->biases[l]));
		newline();
	}
}

void ffn_dump_propagation(FFNPropagationPool* prpool) {
	info("Preactivations");
	for (int l = 0; l < (int)prpool->base.layer_cnt; l++) {
		vec_dump(&(prpool->preactivations[l]));
		newline();
	}
	info("Activations");
	for (int l = 0; l < (int)prpool->base.layer_cnt; l++) {
		vec_dump(&(prpool->activations[l]));
		newline();
	}
}

void ffn_dump_gradient(FFNGradientPool* gpool) {
	info("Weight gradients");
	for (int l = 0; l < ((int)(gpool->base.layer_cnt)-1); l++) {
		info("Layer %d:", l);
		matrix_dump_raw(&(gpool->gradient_w[l]));
		newline();
	}
	newline();
	info("Bias gradients");
	for (int l = 0; l < ((int)(gpool->base.layer_cnt)); l++) {
		info("Layer %d:", l);
		vec_dump(&(gpool->gradient_b[l]));
		newline();
	}
}
