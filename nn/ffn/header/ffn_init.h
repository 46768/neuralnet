#ifndef FFN_INIT_H
#define FFN_INIT_H

#include <stdlib.h>

#include "ffn_type.h"

// Creation
FFN* ffn_init(); // Create a feed forward network

// Layer initialization
void ffn_init_dense(FFN*, size_t); // Push a dense (fully connected) layer
void ffn_init_params(FFN*); // Finalize a network's layer

// Memory management
void ffn_deallocate(FFN*); // Deallocate a network

#endif
