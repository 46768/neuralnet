#include "optimizer.h"

OptimizerFn resolve_optimizer_fn(OptimizerFnEnum fn) {
	switch(fn) {
		case GD:
			return nn_gradient_descent;
		case Momentum:
			return nn_momentun_optimize;
		default:
			return NULL;
	}
}
