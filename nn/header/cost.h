#ifndef NN_COST_H
#define NN_COST_H

#include "vector.h"

typedef enum { MSE } CostEnum;

typedef float (*CostFn)(Vector*, Vector*);
typedef void (*CostFnD)(Vector*, Vector*, Vector*);

CostFn cost_resolve(CostEnum);
CostFnD cost_d_resolve(CostEnum);

float cost_mse(Vector*, Vector*);
void cost_mse_d(Vector*, Vector*, Vector*);

float cost_ccel(Vector*, Vector*);
void cost_ccel_d(Vector*, Vector*, Vector*);

#endif
