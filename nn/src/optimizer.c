#include "optimizer.h"

#include <string.h>

#include "allocator.h"

size_t resolve_optimizer_config_size(OptimizerTypeEnum type) {
	switch (type) {
		case GD:
			return sizeof(GradientDescentConfig);
		case Momentum:
			return sizeof(MomentumConfig);
		default:
			return 0;
	}
}

OptimizerFn resolve_optimizer(OptimizerTypeEnum type) {
	switch (type) {
		case GD:
			return nn_gradient_descent;
		case Momentum:
			return nn_momentum_optimize;
		default:
			return 0;
	}
}

OptimizerFn resolve_optimizer_finalizer(OptimizerTypeEnum type) {
	switch (type) {
		case GD:
			return nn_gradient_descent_finalize;
		case Momentum:
			return nn_momentum_optimize_finalize;
		default:
			return 0;
	}
}

// Optimizer builder

Optimizer* nn_build_optimizer(OptimizerTypeEnum type, void* config) {
	Optimizer* optimizer = (Optimizer*)allocate(sizeof(Optimizer));
	size_t config_size = resolve_optimizer_config_size(type);

	optimizer->config = allocate(config_size);
	optimizer->fn = resolve_optimizer(type);
	optimizer->finalize = resolve_optimizer_finalizer(type);
	optimizer->type = type;

	memcpy(optimizer->config, config, config_size);

	return optimizer;
}
