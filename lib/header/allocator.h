#ifndef COM_ALLOCATOR_H
#define COM_ALLOCATOR_H

#include <stdlib.h>

void* allocate(size_t);
void* reallocate(void*, size_t);
void deallocate(void*);

#endif
