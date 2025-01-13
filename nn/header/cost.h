#ifndef NN_COST_H
#define NN_COST_H

#include "vector.h"

// Cost Function Type
typedef float(*CostFn)(Vector*, Vector*); // Cost Function
typedef Vector*(*CostFnD)(Vector*, Vector*); // Cost Function Derivative

// Cost Function Enum
typedef enum {
	MSE,
	CEL,
} CostFnEnum;

// Cost Function Resolver
CostFn resolve_cost_fn(CostFnEnum);
CostFnD resolve_cost_fn_d(CostFnEnum);

// Mean Squared Error
float nn_mse(Vector*, Vector*);
Vector* nn_mse_d(Vector*, Vector*);

// Cross Entropy Loss
float nn_cel(Vector*, Vector*);
Vector* nn_cel_d(Vector*, Vector*);

#endif
