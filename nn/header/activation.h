#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H

#include "vector.h"

// Activation Function Type
typedef Vector*(*activation_fn)(Vector*); // Activation Function
typedef Vector*(*activation_fn_d)(Vector*); // Activation Function Derivative

// Activation Function Enum
typedef enum {
	ReLU,
	Sigmoid
} ActivationFN;

// Activation Function Resolver
activation_fn resolve_activation_fn(ActivationFN);
activation_fn_d resolve_activation_fn_d(ActivationFN);

// ReLU
Vector* nn_relu(Vector*);
Vector* nn_relu_d(Vector*);

// Sigmoid
Vector* nn_sigmoid(Vector*);
Vector* nn_sigmoid_d(Vector*);

#endif
