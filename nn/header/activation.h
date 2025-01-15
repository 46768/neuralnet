#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H

#include "vector.h"

// Activation Function Type
typedef void(*ActivationFn)(Vector*, Vector*); // Activation Function
typedef void(*ActivationFnD)(Vector*, Vector*); // Activation Function Derivative

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
void nn_none_fn(Vector*, Vector*);
void nn_none_fn_d(Vector*, Vector*);

// ReLU
void nn_relu(Vector*, Vector*);
void nn_relu_d(Vector*, Vector*);

// Sigmoid
void nn_sigmoid(Vector*, Vector*);
void nn_sigmoid_d(Vector*, Vector*);

// Softmax
void nn_softmax(Vector*, Vector*);

#endif
