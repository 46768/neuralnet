#include "allocator.h"

#include <stdlib.h>

#include "logger.h"

void* allocate(size_t size) {
	void* ptr = malloc(size);
	if (ptr == NULL) {
		fatal("Failed to allocate a pointer sized %zu bytes", size);
		exit(1);
	}
	return ptr;
}

void* reallocate(void* ptr, size_t size) {
	if (ptr == NULL) {
		fatal("Provided pointer is NULL");
		exit(1);
	}
	void* ptr_new = realloc(ptr, size);
	if (ptr_new == NULL) {
		fatal("Failed to reallocate a pointer sized %zu bytes", size);
		exit(1);
	}
	return ptr_new;
}

void deallocate(void* ptr) {
	if (ptr != NULL) {
		free(ptr);
		ptr = NULL;
	}
}
