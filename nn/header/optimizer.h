#ifndef NN_OPTIMIZER_H
#define NN_OPTIMIZER_H

#include <stdlib.h>

// Optimizer Function Type (float* buf, float* grad, size_t grad_size, void* self)
typedef void(*OptimizerFn)(float*, float*, size_t, void*);

// Batch Type Enum
typedef enum {
	Stochastic=0,
	MiniBatch,
	FullBatch
} BatchTypeEnum;

// Generic Optimizer Struct
typedef struct {
	void* config;
	float* buf;
	size_t batch_size;
	OptimizerFn fn;
	OptimizerFn finalize;
} Optimizer;

// Optimizers

typedef struct {
	size_t batch_size;
} GradientDescentConfig;
Optimizer* nn_gradient_descent_init();
void nn_gradient_descent(float*, float*, size_t, void*);
void nn_gradient_descent_finalize(float*, float*, size_t, void*);

typedef struct {
	size_t batch_size;
	float velocity_coef;
} MomentumConfig;
Optimizer* nn_momentum_optimize_init(float);
void nn_momentum_optimize(float*, float*, size_t, void*);
void nn_momentum_optimize_finalize(float*, float*, size_t, void*);

#endif
