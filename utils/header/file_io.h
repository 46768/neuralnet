#ifndef COM_FILE_READER_H
#define COM_FILE_READER_H

#include <stdio.h>

typedef struct {
	unsigned long size;
	char* filename;
	FILE* file_pointer;
} FileData;

FileData* get_file_read(char*);
FileData* get_file_write(char*);
int close_file(FileData*);

#endif
