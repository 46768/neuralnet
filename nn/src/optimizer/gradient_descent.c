#include "optimizer.h"

#include <string.h>

#include "allocator.h"

#ifdef SIMD_AVX
#include "avx.h"
#include "avxmm.h"
#endif

Optimizer* nn_gradient_descent_init() {
	Optimizer* optimizer = (Optimizer*)allocate(sizeof(Optimizer));
	optimizer->config = allocate(sizeof(GradientDescentConfig));
	optimizer->fn = nn_gradient_descent;
	optimizer->finalize = nn_gradient_descent_finalize;
	optimizer->type = GD;

	return optimizer;
}
void nn_gradient_descent(float* buffer, float* gradient, size_t gradient_size, void* _) {
	size_t float_cnt = gradient_size / sizeof(float);

#ifdef SIMD_AVX
	for (int i = 0; i < (int)float_cnt; i+=8) {
		avx_add(&(buffer[i]), &(gradient[i]), &(buffer[i]));
	}
#else
	for (int i = 0; i < (int)float_cnt; i++) {
		buffer[i] += gradient[i];
	}
#endif
}

void nn_gradient_descent_finalize(float* buffer, float* gradient, size_t gradient_size, void* config) {
	size_t float_cnt = gradient_size / sizeof(float);
	float grad_coef = (float)1 / ((GradientDescentConfig*)(config))->batch_size;

#ifdef SIMD_AVX
	AVX256 vgrad_coef = avxmm256_load_single_ptr(grad_coef);
	for (int i = 0; i < (int)float_cnt; i+=8) {
		avxmm256_unload_ptr(avxmm256_mul(avxmm256_load_ptr(&(buffer[i])), vgrad_coef), &(buffer[i]));
	}
#else
	for (int i = 0; i < (int)float_cnt; i++) {
		buffer[i] *= grad_coef;
	}
#endif

	memcpy(gradient, buffer, gradient_size);
	memset(buffer, 0, gradient_size);
}
