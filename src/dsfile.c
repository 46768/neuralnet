#include <stdio.h>

#include "vector.h"
#include "file_io.h"
#include "datasets.h"

int main() {
	FileData *file = file_get_read("train_dataset.bin");
	Dataset *dataset = dataset_file(file->file_pointer);
	//uint32_t ds_size = dataset->size;
	uint32_t ds_size = dataset->size < 10 ? dataset->size : 10; // Only get first 10 entries

	Vector *in = dataset->input;
	Vector *out = dataset->target;

	printf("Dataset Metadata: size %d isize %d osizs %d\n", dataset->size, in->size, out->size);

	for (uint32_t i = 0; i < ds_size; i++) {
		vec_print(out+i);
	}

	dataset_free(dataset);
	file_close(file);

	return 0;
}
