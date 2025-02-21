#ifndef THREADING_H
#define THREADING_H

#include <pthread.h>

typedef struct {
	size_t thread_count;
	pthread_t* thread_ptr;
} ThreadPool;

int threading_new(pthread_t*, const pthread_attr_t*, void*(*)(void*), void*);

#endif
