#include "activation.h"
#include "cost.h"
#include "initer.h"

ActivationFn activation_resolve(ActivationEnum enm) {
	switch (enm) {
		case None:
			return activation_none;
		case Sigmoid:
			return activation_sigmoid;
	}

	return activation_none;
}

ActivationFnD activation_d_resolve(ActivationEnum enm) {
	switch (enm) {
		case None:
			return activation_none_d;
		case Sigmoid:
			return activation_sigmoid_d;
	}

	return activation_none_d;
}

CostFn cost_resolve(CostEnum enm) {
	switch (enm) {
		case MSE:
			return cost_mse;
	}

	return cost_mse;
}

CostFnD cost_d_resolve(CostEnum enm) {
	switch (enm) {
		case MSE:
			return cost_mse_d;
	}

	return cost_mse_d;
}

InitFn initer_resolve(IniterEnum enm) {
	switch (enm) {
		case Zero:
			return initer_zero;
		case RandomEN2:
			return initer_random_en2;
		case He:
			return initer_he;
		case Xavier:
			return initer_xavier;
	}

	return initer_zero;
}
