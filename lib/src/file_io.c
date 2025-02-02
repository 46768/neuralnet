#include "file_io.h"

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>

#include "allocator.h"
#include "logger.h"

FileData* _get_file(char* filename, char* mode) {
	char* fullname = (char*)allocate(strlen(PROJECT_PATH "/data/")+strlen(filename)+2);
	strcat(fullname, PROJECT_PATH "/data/");
	strcat(fullname, filename);
	FileData* file_read = (FileData*)allocate(sizeof(FileData));
	mkdir(PROJECT_PATH "/data", 0777);
	if (access(fullname, F_OK) != 0) {
		 FILE* temp = fopen(fullname, "w");
		 if (temp == NULL) {
			fatal("Failed to open file: %s", fullname);
		 }
		 fclose(temp);
	}
	// Open file
	file_read->file_pointer = fopen(fullname, mode);
	file_read->filename = fullname;

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
	deallocate(file_data->filename);
	deallocate(file_data);
	return 0;
}
