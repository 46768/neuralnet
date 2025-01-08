#include "file_reader.h"

#include <stdio.h>


FileData get_file(char* filename) {
	FileData file_read;
	// Open file
	file_read.file_pointer = fopen(filename, "rb");
	file_read.filename = filename;

	// Get file size
	fseek(file_read.file_pointer, 0l, SEEK_END);
	file_read.size = ftell(file_read.file_pointer);
	fseek(file_read.file_pointer, 0l, SEEK_SET);

	return file_read;
}

int close_file(FileData file_data) {
	if (fclose(file_data.file_pointer) != 0) {
		printf("Failed to close file: %s", file_data.filename);
		return 1;
	}
	return 0;
}
