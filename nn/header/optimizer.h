#ifndef NN_OPTIMIZER_H
#define NN_OPTIMIZER_H

#include "matrix.h"

// Optimizer Function Types
typedef void(*OptimizerFn)(Matrix*, Matrix*); // Optimizer Function

// Cost Function Enum
typedef enum {
	SGD,
	Momentum
} OptimizerFnEnum;

// Gradient Descent
void nn_gradient_descent(Matrix*, Matrix*);

// Momentum Optimizer
void nn_momentun_optimize(Matrix*, Matrix*);

#endif
