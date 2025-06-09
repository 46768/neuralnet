#ifdef SIMD_AVX2
#include <immintrin.h>
#include <string.h>

#include "matrix.h"

// Matrix Operation

void mat_vmul(Matrix *mat, Vector *vec, Vector *res) {
	// Get contexts for the kernel

	get_mat_ctx(mat_ctx, mat);
	get_vec_ctx(vec_ctx, vec);
	get_vec_ctx(res_ctx, res);

	// Compute the multiplication with the kernel

	__m256 v0, v1, v2, v3, v4, v5, v6, v7,
		   m0, m1, m2, m3, m4, m5, m6, m7;
	for (uint32_t x = 0; x < mat_ctx.sy; x+=8) {
		m7 = _mm256_setzero_ps();
		for (uint32_t y = 0; y < mat_ctx.sx; y+=8) {
			v0 = _mm256_set1_ps(vec_idx(vec_ctx, y+0));
			v1 = _mm256_set1_ps(vec_idx(vec_ctx, y+1));
			v2 = _mm256_set1_ps(vec_idx(vec_ctx, y+2));
			v3 = _mm256_set1_ps(vec_idx(vec_ctx, y+3));
			v4 = _mm256_set1_ps(vec_idx(vec_ctx, y+4));
			v5 = _mm256_set1_ps(vec_idx(vec_ctx, y+5));
			v6 = _mm256_set1_ps(vec_idx(vec_ctx, y+6));
			v7 = _mm256_set1_ps(vec_idx(vec_ctx, y+7));

			m0 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x+0, y+0));
			m1 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x+0, y+1));
			m2 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x+0, y+2));
			m3 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x+0, y+3));
			m4 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x+0, y+4));
			m5 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x+0, y+5));
			m6 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x+0, y+6));

			m7 = _mm256_fmadd_ps(v0, m0, m7);
			m7 = _mm256_fmadd_ps(v1, m1, m7);
			m7 = _mm256_fmadd_ps(v2, m2, m7);
			m7 = _mm256_fmadd_ps(v3, m3, m7);
			m7 = _mm256_fmadd_ps(v4, m4, m7);
			m7 = _mm256_fmadd_ps(v5, m5, m7);
			m7 = _mm256_fmadd_ps(v6, m6, m7);

			m0 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x+0, y+7));
			m7 = _mm256_fmadd_ps(v7, m0, m7);
		}

		_mm256_store_ps(vec_idx_ptr(res_ctx, x), m7);
	}
}

void mat_fmva(Matrix* mat, Vector* vec, Vector* res) {
	// Get contexts for the kernel

	get_mat_ctx(mat_ctx, mat);
	get_vec_ctx(vec_ctx, vec);
	get_vec_ctx(res_ctx, res);

	// Compute the multiplication with the kernel

	__m256 v0, v1, v2, v3, v4, v5, v6, v7,
		   m0, m1, m2, m3, m4, m5, m6, m7;
	for (uint32_t x = 0; x < mat_ctx.sy; x+=8) {
		m7 = _mm256_load_ps(vec_idx_ptr(res_ctx, x));
		for (uint32_t y = 0; y < mat_ctx.sx; y+=8) {
			v0 = _mm256_set1_ps(vec_idx(vec_ctx, y+0));
			v1 = _mm256_set1_ps(vec_idx(vec_ctx, y+1));
			v2 = _mm256_set1_ps(vec_idx(vec_ctx, y+2));
			v3 = _mm256_set1_ps(vec_idx(vec_ctx, y+3));
			v4 = _mm256_set1_ps(vec_idx(vec_ctx, y+4));
			v5 = _mm256_set1_ps(vec_idx(vec_ctx, y+5));
			v6 = _mm256_set1_ps(vec_idx(vec_ctx, y+6));
			v7 = _mm256_set1_ps(vec_idx(vec_ctx, y+7));

			m0 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x+0, y+0));
			m1 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x+0, y+1));
			m2 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x+0, y+2));
			m3 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x+0, y+3));
			m4 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x+0, y+4));
			m5 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x+0, y+5));
			m6 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x+0, y+6));

			m7 = _mm256_fmadd_ps(v0, m0, m7);
			m7 = _mm256_fmadd_ps(v1, m1, m7);
			m7 = _mm256_fmadd_ps(v2, m2, m7);
			m7 = _mm256_fmadd_ps(v3, m3, m7);
			m7 = _mm256_fmadd_ps(v4, m4, m7);
			m7 = _mm256_fmadd_ps(v5, m5, m7);
			m7 = _mm256_fmadd_ps(v6, m6, m7);

			m0 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x+0, y+7));
			m7 = _mm256_fmadd_ps(v7, m0, m7);
		}

		_mm256_store_ps(vec_idx_ptr(res_ctx, x), m7);
	}
}

void mat_t_vmul(MatrixTranpose* mat, Vector* vec, Vector* res) {
	// Get contexts for the kernel

	get_mat_t_ctx(mat_ctx, mat);
	get_vec_ctx(vec_ctx, vec);
	get_vec_ctx(res_ctx, res);

	// Compute the multiplication with the kernel

	__m256 v0, v1, v2, v3, v4, v5, v6, v7,
		   m0, m1, m2, m3, m4, m5, m6, m7;
	for (uint32_t x = 0; x < mat_ctx.sy; x+=8) {
		m7 = _mm256_setzero_ps();
		for (uint32_t y = 0; y < mat_ctx.sx; y+=8) {
			v0 = _mm256_set1_ps(vec_idx(vec_ctx, y+0));
			v1 = _mm256_set1_ps(vec_idx(vec_ctx, y+1));
			v2 = _mm256_set1_ps(vec_idx(vec_ctx, y+2));
			v3 = _mm256_set1_ps(vec_idx(vec_ctx, y+3));
			v4 = _mm256_set1_ps(vec_idx(vec_ctx, y+4));
			v5 = _mm256_set1_ps(vec_idx(vec_ctx, y+5));
			v6 = _mm256_set1_ps(vec_idx(vec_ctx, y+6));
			v7 = _mm256_set1_ps(vec_idx(vec_ctx, y+7));

			m0 = _mm256_load_ps(mat_dtt_idx_ptr(mat_ctx, x+0, y+0));
			m1 = _mm256_load_ps(mat_dtt_idx_ptr(mat_ctx, x+0, y+1));
			m2 = _mm256_load_ps(mat_dtt_idx_ptr(mat_ctx, x+0, y+2));
			m3 = _mm256_load_ps(mat_dtt_idx_ptr(mat_ctx, x+0, y+3));
			m4 = _mm256_load_ps(mat_dtt_idx_ptr(mat_ctx, x+0, y+4));
			m5 = _mm256_load_ps(mat_dtt_idx_ptr(mat_ctx, x+0, y+5));
			m6 = _mm256_load_ps(mat_dtt_idx_ptr(mat_ctx, x+0, y+6));

			m7 = _mm256_fmadd_ps(v0, m0, m7);
			m7 = _mm256_fmadd_ps(v1, m1, m7);
			m7 = _mm256_fmadd_ps(v2, m2, m7);
			m7 = _mm256_fmadd_ps(v3, m3, m7);
			m7 = _mm256_fmadd_ps(v4, m4, m7);
			m7 = _mm256_fmadd_ps(v5, m5, m7);
			m7 = _mm256_fmadd_ps(v6, m6, m7);

			m0 = _mm256_load_ps(mat_dtt_idx_ptr(mat_ctx, x+0, y+7));
			m7 = _mm256_fmadd_ps(v7, m0, m7);
		}

		_mm256_store_ps(vec_idx_ptr(res_ctx, x), m7);
	}
}

// Vector Operation (Matrix return)

void vec_crmul(Vector* c_vec, Vector* r_vec, Matrix* mat) {
}

#endif
