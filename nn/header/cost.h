#ifndef NN_COST_H
#define NN_COST_H

#include "vector.h"

// Cost Function Type
typedef float(*CostFn)(Vector*, Vector*); // Cost Function
typedef void(*CostFnD)(Vector*, Vector*, Vector*); // Cost Function Derivative

// Cost Function Enum
typedef enum {
	MSE,
	CCE,
	BCE,
} CostFnEnum;

// Cost Function Resolver
CostFn resolve_cost_fn(CostFnEnum);
CostFnD resolve_cost_fn_d(CostFnEnum);

// Mean Squared Error
float nn_mse(Vector*, Vector*);
void nn_mse_d(Vector*, Vector*, Vector*);

// Categorical Cross Entropy Loss
float nn_ccel(Vector*, Vector*);
void nn_ccel_d(Vector*, Vector*, Vector*);

// Binary Cross Entropy Loss
float nn_bcel(Vector*, Vector*);
void nn_bcel_d(Vector*, Vector*, Vector*);

#endif
