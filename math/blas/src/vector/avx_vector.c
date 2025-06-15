#include "vector.h"

#include <immintrin.h>

void vec_cadd(Vector *vec1, Vector *vec2, float mcoef) {
    get_vec_ctx(v1, vec1);
    get_vec_ctx(v2, vec2);

    __m256 v1d, v2d, vcoef;
    vcoef = _mm256_set1_ps(mcoef);
    for (uint32_t i = 0; i < v1.size; i += 8) {
        v1d = _mm256_load_ps(vec_idx_ptr(v1, i));
        v2d = _mm256_load_ps(vec_idx_ptr(v2, i));
        v2d = _mm256_mul_ps(v2d, vcoef);
        v1d = _mm256_add_ps(v1d, v2d);
        _mm256_store_ps(vec_idx_ptr(v1, i), v1d);
    }
}

void vec_mul(Vector *vec1, Vector *vec2) {
    get_vec_ctx(v1, vec1);
    get_vec_ctx(v2, vec2);

    __m256 v1d, v2d;
    for (uint32_t i = 0; i < v1.size; i += 8) {
        v1d = _mm256_load_ps(vec_idx_ptr(v1, i));
        v2d = _mm256_load_ps(vec_idx_ptr(v2, i));
        v1d = _mm256_mul_ps(v1d, v2d);
        _mm256_store_ps(vec_idx_ptr(v1, i), v1d);
    }
}
