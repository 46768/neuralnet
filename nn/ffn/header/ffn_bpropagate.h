#ifndef FFN_BPROPAGATE_H
#define FFN_BPROPAGATE_H

#include "ffn_mempool.h"

#include "vector.h"

float ffn_get_param_change(FFNParams*, FFNParameterPool*, FFNPropagationPool*, FFNGradientPool*, FFNIntermediatePool*, Vector*);
void ffn_apply_gradient(FFNParams*, FFNParameterPool*, FFNGradientPool*, float);

#endif
