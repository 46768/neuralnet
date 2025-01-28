#ifndef NN_GENERATOR_H
#define NN_GENERATOR_H

#include "vector.h"

// Linear Regression Generator
void generate_linear_regs(int, int, float, float, Vector***, Vector***);

// Binary XOR Generator
void generate_xor(int*, int*, Vector***, Vector***);

#endif
