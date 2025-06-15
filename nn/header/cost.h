#ifndef NN_COST_H
#define NN_COST_H

typedef enum { MSE } CostEnum;

typedef void (*CostFn)();
typedef void (*CostFnD)();

CostFn cost_resolve(CostEnum);
CostFnD cost_d_resolve(CostEnum);

#endif
