#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H

#include "vector.h"

typedef enum { None } ActivationEnum;

typedef void (*ActivationFn)(Vector*, Vector*);
typedef void (*ActivationFnD)(Vector*, Vector*);

ActivationFn activation_resolve(ActivationEnum);
ActivationFnD activation_d_resolve(ActivationEnum);

void activation_none(Vector*, Vector*);
void activation_none_d(Vector*, Vector*);

#endif
