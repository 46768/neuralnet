#include "datasets.h"

#include <stdlib.h>

#include "vector.h"

#ifdef SIMD_AVX
#include "avx.h"
#define data_alloc(s) avx_allocate(s)
#define data_pad(s) ((s + 63) & ~63) - s
#else
#define data_alloc(s) malloc(s)
#define data_pad(s) 0
#endif

void _dataset_allocate(uint32_t pair_cnt, uint32_t in_size,
                           uint32_t out_size, Dataset *dataset) {
    uint64_t mdata_size = pair_cnt * sizeof(Vector) * 2;
    uint32_t padding = data_pad(mdata_size);
    uint64_t data_size =
        calc_vec_size(in_size) +
        calc_vec_size(out_size); // Size of a pair (count of floats)
    data_size *=
        pair_cnt *
        sizeof(float); // Multiply by amount of pairs then convert to bytes
    void *dptr = data_alloc(mdata_size + padding + data_size);

	Vector* input_ptr = (Vector*)dptr;
	Vector* target_ptr = input_ptr + pair_cnt;

    float *d_ptr = (float *)((((char *)(target_ptr + pair_cnt)) + padding));

    uint64_t d_offset = 0;

    for (uint32_t l = 0; l < pair_cnt; l++) {
        vec_init(in_size, d_ptr + d_offset, input_ptr + l);

        d_offset += calc_vec_size(in_size);
    }

    for (uint32_t l = 0; l < pair_cnt; l++) {
        vec_init(out_size, d_ptr + d_offset, target_ptr + l);

        d_offset += calc_vec_size(out_size);
    }

	dataset->data = dptr;
	dataset->input = input_ptr;
	dataset->target = target_ptr;
	dataset->size = pair_cnt;
}

Dataset *dataset_init(uint32_t pair_cnt, uint32_t input_size, uint32_t target_size) {
	Dataset *dataset = malloc(sizeof(Dataset));
	_dataset_allocate(pair_cnt, input_size, target_size, dataset);

	return dataset;
}

void dataset_free(Dataset *dataset) {
	free(dataset->data);
	free(dataset);
}

Dataset *dataset_linear(int32_t lower_bound, int32_t upper_bound, int32_t slope,
                    int32_t y_intercept) {
	uint32_t point_count = upper_bound - lower_bound + 1;
	Dataset *dataset = dataset_init(point_count, 1, 1);

	Vector *input_vector = dataset->input;
	Vector *output_vector = dataset->target;
	for (uint32_t i = 0; i < point_count; i++) {
		get_vec_ctx(in_vec, input_vector+i);
		get_vec_ctx(out_vec, output_vector+i);

		int32_t x = lower_bound + i;
		in_vec.data[0] = x;
		out_vec.data[0] = (slope * x) + y_intercept;
	}

	return dataset;
}
