#include "file_io.h"

#include <stdio.h>
#include <unistd.h>

#include "allocator.h"

FileData* _get_file(char* filename, char* mode) {
	FileData* file_read = (FileData*)allocate(sizeof(FileData));
	if (access(filename, F_OK) != 0) {
		 FILE* temp = fopen(filename, "w");
		 fclose(temp);
	}
	// Open file
	file_read->file_pointer = fopen(filename, mode);
	file_read->filename = filename;

	// Get file size
	fseek(file_read->file_pointer, 0l, SEEK_END);
	file_read->size = ftell(file_read->file_pointer);
	fseek(file_read->file_pointer, 0l, SEEK_SET);

	return file_read;
}

FileData* get_file_read(char* filename) {
	return _get_file(filename, "rb");
}

FileData* get_file_write(char* filename) {
	return _get_file(filename, "wb");
}

int close_file(FileData* file_data) {
	if (fclose(file_data->file_pointer) != 0) {
		printf("Failed to close file: %s", file_data->filename);
		return 1;
	}
	return 0;
}
