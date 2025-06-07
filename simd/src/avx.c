#include "avx.h"

#include <stdlib.h>

void* avx_allocate(uint32_t size) {
	void* ptr;
	posix_memalign(&ptr, 64, (size+31)&~31);
	return ptr;
}
