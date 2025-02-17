#include "ffn_util.h"

#include "logger.h"

#include "matrix.h"
#include "vector.h"

#include "ffn_mempool.h"

void ffn_dump_param(FFNParameterPool* papool) {
	info("Weights");
	for (int l = 0; l < ((int)(papool->layer_cnt)-1); l++) {
		info("Layer %d:", l);
		matrix_dump_raw(&(papool->weights[l]));
		newline();
	}
	newline();
	info("Biases");
	for (int l = 0; l < ((int)(papool->layer_cnt)-1); l++) {
		info("Layer %d:", l);
		vec_dump(&(papool->biases[l]));
		newline();
	}
}
