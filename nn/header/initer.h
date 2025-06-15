#ifndef NN_INITER_H
#define NN_INITER_H

typedef enum { Zero } IniterEnum;

typedef void (*InitFn)();

InitFn initer_resolve(IniterEnum);

#endif
