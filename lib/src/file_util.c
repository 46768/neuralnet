#include "file_util.h"

#include "allocator.h"

int get_section(FILE* file, char** str_buf, size_t* max_size, char delimiter) {
	int buf_idx = 0;
	char char_buf;

	while ((char_buf = fgetc(file)) != EOF && char_buf != delimiter && char_buf != '\n') {
		(*str_buf)[buf_idx++] = char_buf;

		if (buf_idx >= *max_size) {
			*max_size *= 2;
			*str_buf = (char*)reallocate(*str_buf, *max_size);
		}
	}
	(*str_buf)[buf_idx] = '\0';
	if (char_buf == EOF && buf_idx == 0) return -1;
	return char_buf == delimiter ? 0 : 1;
}
