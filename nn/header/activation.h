#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H

#include "vector.h"

// Activation Function Type
typedef Vector*(*ActivationFn)(Vector*); // Activation Function
typedef Vector*(*ActivationFnD)(Vector*); // Activation Function Derivative

// Activation Function Enum
typedef enum {
	ReLU,
	Sigmoid,
	None
} ActivationFNEnum;

// Activation Function Resolver
ActivationFn resolve_activation_fn(ActivationFNEnum);
ActivationFnD resolve_activation_fn_d(ActivationFNEnum);

// None
Vector* nn_none_fn(Vector*);
Vector* nn_none_fn_d(Vector*);

// ReLU
Vector* nn_relu(Vector*);
Vector* nn_relu_d(Vector*);

// Sigmoid
Vector* nn_sigmoid(Vector*);
Vector* nn_sigmoid_d(Vector*);

// Softmax
Vector* nn_softmax(Vector*);

#endif
