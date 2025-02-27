/** \file */
#ifndef COM_ALLOCATOR_H
#define COM_ALLOCATOR_H

#include <stdlib.h>

/**
 * \brief malloc wrapper with pointer check
 *
 * malloc wrapper that NULL check the pointer
 * created, fatal if fails
 *
 * \param size Size of the memory allocated in bytes
 *
 * \return A pointer pointing to the memory allocated
 */
void* allocate(size_t);

/**
 * \brief calloc wrapper with pointer check
 *
 * calloc wrapper that NULL check the pointer
 * created, fatal if fails
 *
 * \param size Amount of element
 * \param t_size Size of an element
 *
 * \return A pointer pointing to memory sized size*t_size bytes
 */
void* callocate(size_t, size_t);

/**
 * \brief realloc wrapper with pointer check
 *
 * realloc wrapper that NULL check the pointer
 * before realloc, and another NULL check for
 * the new pointer, fatal if either check
 * failed
 *
 * \param ptr Pointer to reallocate
 * \param size Size for the new pointer
 *
 * \return A pointer pointing to new memory with the new size and same data
 */
void* reallocate(void*, size_t);

/**
 * \brief free wrapper with pointer check
 *
 * free wraper that perform NULL check before freeing
 * the pointer provided
 *
 * \param ptr Memory pointer to free
 */
void deallocate(void*);

#endif
