#ifndef FFN_INIT_H
#define FFN_INIT_H

#include <stdlib.h>

#include "ffn_type.h"

// Creation
FFN* ffn_init(); // Create a feed forward network

// Network Settings
void ffn_set_cost_fn(FFN*, CostFnEnum); // Set network's cost function

// Layer initialization
void ffn_init_dense(FFN*, size_t, ActivationFNEnum); // Push a dense (fully connected) layer
void ffn_init_passthru(FFN*, ActivationFNEnum); // Push a pass through layer
void ffn_init_params(FFN*); // Finalize a network's layer

// Memory management
void ffn_deallocate(FFN*); // Deallocate a network

#endif
