#ifndef NN_INITER_H
#define NN_INITER_H

#include <stdlib.h>

// Parameter Initializer Function Type
typedef float(*Initer)(size_t);

// Initializer Function Type Enum
typedef enum {
	Zero,
	He,
	Xavier,
	RandomEN2,
	RandomE0,
} IniterEnum;

// Initializer Function Resolver
Initer resolve_initer(IniterEnum);

float zero_init(size_t); // Initialize to 0.0f
float he_init(size_t); // Initialize with He Initialization
float xavier_init(size_t); // Initialize with Xavier Initialization
float random_en2_init(size_t); // Initialize randomly in range of -0.01 to 0.01
float random_e0_init(size_t); // Initialize randomly in range of -1 to 1

#endif
