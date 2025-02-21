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

// Optimizer Type Enum
typedef enum {
	GD,
	Momentum
} OptimizerTypeEnum;

// Generic Optimizer Struct
typedef struct {
	void* config;
	float* buf;
	size_t batch_size;
	OptimizerTypeEnum type;
	OptimizerFn fn;
	OptimizerFn finalize;
} Optimizer;

// Optimizer resolvers

size_t resolve_optimizer_config_size(OptimizerTypeEnum);
OptimizerFn resolve_optimizer(OptimizerTypeEnum);
OptimizerFn resolve_optimizer_finalizer(OptimizerTypeEnum);

// Optimizer builder

Optimizer* nn_build_optimizer(OptimizerTypeEnum, void*);

// Optimizers

/**
 * \struct GradientDescentConfig
 * \brief Gradient descent configuration
 */
typedef struct {
	size_t batch_size; /**< Batch size */
} GradientDescentConfig;
Optimizer* nn_gradient_descent_init();
void nn_gradient_descent(float*, float*, size_t, void*);
void nn_gradient_descent_finalize(float*, float*, size_t, void*);

/**
 * \struct MomentumConfig
 * \brief Momentum configuration
 */
typedef struct {
	size_t batch_size; /**< Batch size */
	float velocity_coef; /**< Velocity/Momentum term */
} MomentumConfig;
Optimizer* nn_momentum_optimize_init(float);
void nn_momentum_optimize(float*, float*, size_t, void*);
void nn_momentum_optimize_finalize(float*, float*, size_t, void*);

#endif
