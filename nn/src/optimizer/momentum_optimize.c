#include "optimizer.h"

#include <string.h>

#include "allocator.h"

#ifdef SIMD_AVX
#include "avx.h"
#include "avxmm.h"
#endif

Optimizer* nn_momentum_optimize_init(float velocity_coef) {
	Optimizer* optimizer = (Optimizer*)allocate(sizeof(Optimizer));
	optimizer->config = allocate(sizeof(MomentumConfig));
	((MomentumConfig*)(optimizer->config))->velocity_coef = velocity_coef;
	optimizer->fn = nn_momentum_optimize;
	optimizer->finalize = nn_momentum_optimize_finalize;
	optimizer->type = Momentum;

	return optimizer;
}
void nn_momentum_optimize(float* buffer, float* gradient, size_t gradient_size, void* self) {
	size_t float_cnt = gradient_size / sizeof(float);
	float momentum_coef = ((MomentumConfig*)(((Optimizer*)(self))->config))->velocity_coef;
	float inverse_coef = 1 - momentum_coef;
#ifdef SIMD_AVX
	AVX256 vmomentum_coef = avxmm256_load_single_ptr(momentum_coef);
	AVX256 vinverse_coef = avxmm256_load_single_ptr(inverse_coef);

	for (int i = 0; i < (int)float_cnt; i+= 8) {
		avxmm256_unload_ptr(
				avxmm256_madd(
					avxmm256_load_ptr(&(buffer[i])),
					vmomentum_coef,
					avxmm256_mul(
						avxmm256_load_ptr(&(gradient[i])),
						vinverse_coef
						)
					)
				, &(buffer[i]));
	}
#else
	for (int i = 0; i < (int)float_cnt; i++) {
		buffer[i] *= momentum_coef;
		buffer[i] += gradient[i] * inverse_coef;
	}
#endif
}

void nn_momentum_optimize_finalize(float* buffer, float* gradient, size_t gradient_size, void* self) {
	memcpy(gradient, buffer, gradient_size);
}
