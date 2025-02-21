#include "ffn_statistic.h"

#include <stdio.h>

void ffn_export_gradient(FFNModel* model, FileData* datafile) {
	FILE* fptr = datafile->file_pointer;
	size_t float_cnt = model->gpool->base.data_size / sizeof(float);
	float* float_ptr = model->gpool->base.dptr;

	for (int i = 0; i < (int)float_cnt; i++) {
		fprintf(fptr, "%.10f,", float_ptr[i]);
	}
	fflush(fptr);
}
