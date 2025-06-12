#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H

typedef enum {
	None
} ActivationEnum;

typedef void(*ActivationFn)();
typedef void(*ActivationFnD)();

ActivationFn activation_resolve(ActivationEnum);
ActivationFnD activation_d_resolve(ActivationEnum);

#endif
