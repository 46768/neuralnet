#ifndef NN_INITER_H
#define NN_INITER_H

#include <stdint.h>

typedef enum {
	Zero,
	RandomEN2,
	He,
	Xavier,
} IniterEnum;

typedef float (*InitFn)(uint32_t);

InitFn initer_resolve(IniterEnum);

float initer_zero(uint32_t);
float initer_random_en2(uint32_t);
float initer_he(uint32_t);
float initer_xavier(uint32_t);


#endif
