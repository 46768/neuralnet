#include "datasets.h"

#include <stdlib.h>

#include "vector.h"

Dataset* dataset_init() {
	Dataset *ds = malloc(sizeof(Dataset));

	return ds;
}

uint64_t _dataset_allocate(uint32_t pair_cnt, uint32_t i_size, uint32_t o_size, Dataset *ds) {
	uint64_t mdata_size = pair_cnt * sizeof(Vector) * 2;

	uint64_t data_size = calc_vec_size(i_size) + calc_vec_size(o_size);
	data_size *= pair_cnt*sizeof(float);

	return pair_cnt * sizeof(Vector);
}

void dataset_linear(int32_t lower_b, int32_t upper_b, int32_t m, int32_t b, Dataset *ds) {

}
