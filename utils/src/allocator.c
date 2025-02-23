#include "allocator.h"

#include <stdlib.h>

#include "logger.h"

#define _check_ptr(ptr, size) if(ptr==NULL){fatal("Failed to allocate pointer sized %zu bytes",size);}

void* allocate(size_t size) {
	void* ptr = malloc(size);
	_check_ptr(ptr, size);
	return ptr;
}

void* callocate(size_t size, size_t t_size) {
	void* ptr = calloc(size, t_size);
	_check_ptr(ptr, size);
	return ptr;
}

void* reallocate(void* ptr, size_t size) {
	if (ptr == NULL) {fatal("Provided pointer is NULL");}
	void* ptr_new = realloc(ptr, size);
	_check_ptr(ptr, size);
	return ptr_new;
}

void deallocate(void* ptr) {
	if (ptr != NULL) {
		free(ptr);
	}
}
