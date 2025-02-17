#ifndef NN_OPTIMIZER_H
#define NN_OPTIMIZER_H

// Optimizer Function Types
typedef void(*OptimizerFn)(float*, float*, void*); // Optimizer Function

// Batch Type Enum
typedef enum {
	Stochastic=0,
	MiniBatch,
	FullBatch
} BatchTypeEnum;
// Optimizer Function Enum
typedef enum {
	GD=0,
	Momentum
} OptimizerFnEnum;

// Resolvers

OptimizerFn resolve_optimizer_fn(OptimizerFnEnum);

// Optimizers

typedef struct {

} GradientDescentExtra;
void nn_gradient_descent(float*, float*, void*);

typedef struct {

} MomentumExtra;
void nn_momentun_optimize(float*, float*, void*);

#endif
