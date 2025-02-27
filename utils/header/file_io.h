/** \file */
#ifndef COM_FILE_READER_H
#define COM_FILE_READER_H

#include <stdio.h>

/**
 * \struct FileData
 * \brief A FILE* wrapper with extra data
 */
typedef struct {
	unsigned long size; /**< File size */
	char* filename; /**< File name */
	FILE* file_pointer; /**< File pointer */
} FileData;

FileData* get_file_read(char*);
FileData* get_file_write(char*);
int close_file(FileData*);

#endif
