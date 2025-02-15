#ifndef FFN_BPROPAGATE_H
#define FFN_BPROPAGATE_H

#include "ffn_type.h"
#include "ffn_mempool.h"

#include "vector.h"

float ffn_get_param_change(FFN*, FFNMempool*, Vector*); // Back propagation
void ffn_apply_gradient(FFN*, FFNGradientPool*, float);

#endif
