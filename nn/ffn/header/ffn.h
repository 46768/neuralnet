#ifndef FFN_H
#define FFN_H

#include "booltype.h"

#include "vector.h"

#include "activation.h"
#include "cost.h"
#include "initer.h"
#include "optimizer.h"

#include "ffn_mempool.h"
#include "ffn_type.h"

typedef struct {
	FFN* nn;
	FFNMempool* pool;
	OptimizerFnEnum optimizer;
	booltype immutable;
} FFNModel;

// Creation

FFNModel* ffn_new_model();

// Memory Management

void ffn_deallocate_model(FFNModel*);

// Initialization

void ffn_add_dense(FFNModel*, size_t, ActivationFNEnum, IniterEnum, IniterEnum);
void ffn_add_passthrough(FFNModel*, ActivationFNEnum);
void ffn_set_cost_fn(FFNModel*, CostFnEnum);
void ffn_set_optimizer(FFNModel*, OptimizerFnEnum);
void ffn_finalize(FFNModel*);

// Running and Training

Vector* ffn_run(FFNModel*, Vector*);
float ffn_train(FFNModel*, Vector**, Vector**, size_t, float, int);

#endif
