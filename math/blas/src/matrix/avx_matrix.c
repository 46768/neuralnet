#ifdef SIMD_AVX
#include <immintrin.h>
#include <string.h>

#include "matrix.h"

// Matrix Operation

static inline void _mat_vmul_kernel(MatrixCtx mat, Vector vec_ctx,
		Vector res_ctx, uint32_t kx_offset, uint32_t ky_offset) {
	__m256 v0 = _mm256_set1_ps(vec_idx(vec_ctx, kx_offset+0));
	__m256 v1 = _mm256_set1_ps(vec_idx(vec_ctx, kx_offset+1));
	__m256 v2 = _mm256_set1_ps(vec_idx(vec_ctx, kx_offset+2));
	__m256 v3 = _mm256_set1_ps(vec_idx(vec_ctx, kx_offset+3));
	__m256 v4 = _mm256_set1_ps(vec_idx(vec_ctx, kx_offset+4));
	__m256 v5 = _mm256_set1_ps(vec_idx(vec_ctx, kx_offset+5));
	__m256 v6 = _mm256_set1_ps(vec_idx(vec_ctx, kx_offset+6));
	__m256 v7 = _mm256_set1_ps(vec_idx(vec_ctx, kx_offset+7));

	__m256 r = _mm256_setzero_ps();

	__m256 m0 = _mm256_load_ps(mat_t_idx_ptr(mat, kx_offset+0, ky_offset+0));
	__m256 m1 = _mm256_load_ps(mat_t_idx_ptr(mat, kx_offset+1, ky_offset+0));
	__m256 m2 = _mm256_load_ps(mat_t_idx_ptr(mat, kx_offset+2, ky_offset+0));
	__m256 m3 = _mm256_load_ps(mat_t_idx_ptr(mat, kx_offset+3, ky_offset+0));
	__m256 m4 = _mm256_load_ps(mat_t_idx_ptr(mat, kx_offset+4, ky_offset+0));
	__m256 m5 = _mm256_load_ps(mat_t_idx_ptr(mat, kx_offset+5, ky_offset+0));
	__m256 m6 = _mm256_load_ps(mat_t_idx_ptr(mat, kx_offset+6, ky_offset+0));
	__m256 m7 = _mm256_load_ps(mat_t_idx_ptr(mat, kx_offset+7, ky_offset+0));

	__m256 vm0 = _mm256_mul_ps(v0, m0);
	__m256 vm1 = _mm256_mul_ps(v1, m1);
	__m256 vm2 = _mm256_mul_ps(v2, m2);
	__m256 vm3 = _mm256_mul_ps(v3, m3);
	__m256 vm4 = _mm256_mul_ps(v4, m4);
	__m256 vm5 = _mm256_mul_ps(v5, m5);
	__m256 vm6 = _mm256_mul_ps(v6, m6);
	__m256 vm7 = _mm256_mul_ps(v7, m7);

	r = _mm256_add_ps(vm0, r);
	r = _mm256_add_ps(vm1, r);
	r = _mm256_add_ps(vm2, r);
	r = _mm256_add_ps(vm3, r);
	r = _mm256_add_ps(vm4, r);
	r = _mm256_add_ps(vm5, r);
	r = _mm256_add_ps(vm6, r);
	r = _mm256_add_ps(vm7, r);

	_mm256_store_ps(vec_idx_ptr(res_ctx, ky_offset), r);
}

void mat_vmul(Matrix *mat, Vector *vec, Vector *res) {
	// Get contexts for the kernel

	get_mat_ctx(mat_ctx, mat);
	get_vec_ctx(vec_ctx, vec);
	get_vec_ctx(res_ctx, res);

	// Compute the multiplication with the kernel

	for (uint32_t x = 0; x < mat_ctx.rsx; x+=8) {
		for (uint32_t y = 0; y < mat_ctx.sy; y+=8) {
			_mat_vmul_kernel(mat_ctx, vec_ctx, res_ctx, x, y);
		}
	}
}

void mat_fmva(Matrix* mat1, Matrix* mat2, Vector* vec) {
}

void mat_hadamard(Matrix* mat, Vector* vec) {
}

void mat_t_hadamard(Matrix* mat, Vector* vec) {
}

// Vector Operation (Matrix return)

void vec_crmul(Vector* c_vec, Vector* r_vec, Matrix* mat) {
}

#endif
