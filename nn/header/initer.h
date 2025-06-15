#ifndef NN_INITER_H
#define NN_INITER_H

typedef enum { Zero } IniterEnum;

typedef float (*InitFn)();

InitFn initer_resolve(IniterEnum);

float initer_zero();

#endif
