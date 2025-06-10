#ifdef SIMD_AVX
#include <immintrin.h>
#include <string.h>

#include "matrix.h"

// Matrix Operation

#ifndef SIMD_AVX2
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

			m0 = _mm256_mul_ps(v0, m0);
			m7 = _mm256_add_ps(m0, m7);

			m1 = _mm256_mul_ps(v1, m1);
			m7 = _mm256_add_ps(m1, m7);

			m2 = _mm256_mul_ps(v2, m2);
			m7 = _mm256_add_ps(m2, m7);

			m3 = _mm256_mul_ps(v3, m3);
			m7 = _mm256_add_ps(m3, m7);

			m4 = _mm256_mul_ps(v4, m4);
			m7 = _mm256_add_ps(m4, m7);

			m5 = _mm256_mul_ps(v5, m5);
			m7 = _mm256_add_ps(m5, m7);

			m6 = _mm256_mul_ps(v6, m6);
			m7 = _mm256_add_ps(m6, m7);


			m0 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x+0, y+7));
			m0 = _mm256_mul_ps(v7, m0);
			m7 = _mm256_add_ps(m0, m7);
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

			m0 = _mm256_mul_ps(v0, m0);
			m7 = _mm256_add_ps(m0, m7);

			m1 = _mm256_mul_ps(v1, m1);
			m7 = _mm256_add_ps(m1, m7);

			m2 = _mm256_mul_ps(v2, m2);
			m7 = _mm256_add_ps(m2, m7);

			m3 = _mm256_mul_ps(v3, m3);
			m7 = _mm256_add_ps(m3, m7);

			m4 = _mm256_mul_ps(v4, m4);
			m7 = _mm256_add_ps(m4, m7);

			m5 = _mm256_mul_ps(v5, m5);
			m7 = _mm256_add_ps(m5, m7);

			m6 = _mm256_mul_ps(v6, m6);
			m7 = _mm256_add_ps(m6, m7);


			m0 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x+0, y+7));
			m0 = _mm256_mul_ps(v7, m0);
			m7 = _mm256_add_ps(m0, m7);
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

			m0 = _mm256_mul_ps(v0, m0);
			m7 = _mm256_add_ps(m0, m7);

			m1 = _mm256_mul_ps(v1, m1);
			m7 = _mm256_add_ps(m1, m7);

			m2 = _mm256_mul_ps(v2, m2);
			m7 = _mm256_add_ps(m2, m7);

			m3 = _mm256_mul_ps(v3, m3);
			m7 = _mm256_add_ps(m3, m7);

			m4 = _mm256_mul_ps(v4, m4);
			m7 = _mm256_add_ps(m4, m7);

			m5 = _mm256_mul_ps(v5, m5);
			m7 = _mm256_add_ps(m5, m7);

			m6 = _mm256_mul_ps(v6, m6);
			m7 = _mm256_add_ps(m6, m7);

			m0 = _mm256_load_ps(mat_dtt_idx_ptr(mat_ctx, x+0, y+7));
			m0 = _mm256_mul_ps(v7, m0);
			m7 = _mm256_add_ps(m0, m7);
		}

		_mm256_store_ps(vec_idx_ptr(res_ctx, x), m7);
	}
}
#endif

// Vector Operation (Matrix return)

void vec_crmul(Vector* c_vec, Vector* r_vec, Matrix* mat) {
	// Get contexts
	
	get_vec_ctx(cv, c_vec);
	get_vec_ctx(rv, r_vec);
	get_mat_ctx(m, mat);

	__m256 b,
		   r0,r1,r2,r3,r4,r5,r6,r7;
	for (uint32_t i = 0; i < cv.size; i+=8) {
		b = _mm256_load_ps(vec_idx_ptr(cv, i));
		for (uint32_t j = 0; j < rv.size; j+=8) {
			r0 = _mm256_set1_ps(vec_idx(rv, j+0));
			r1 = _mm256_set1_ps(vec_idx(rv, j+1));
			r2 = _mm256_set1_ps(vec_idx(rv, j+2));
			r3 = _mm256_set1_ps(vec_idx(rv, j+3));
			r4 = _mm256_set1_ps(vec_idx(rv, j+4));
			r5 = _mm256_set1_ps(vec_idx(rv, j+5));
			r6 = _mm256_set1_ps(vec_idx(rv, j+6));
			r7 = _mm256_set1_ps(vec_idx(rv, j+7));

			r0 = _mm256_mul_ps(b, r0);
			r1 = _mm256_mul_ps(b, r1);
			r2 = _mm256_mul_ps(b, r2);
			r3 = _mm256_mul_ps(b, r3);
			r4 = _mm256_mul_ps(b, r4);
			r5 = _mm256_mul_ps(b, r5);
			r6 = _mm256_mul_ps(b, r6);
			r7 = _mm256_mul_ps(b, r7);

			_mm256_store_ps(mat_t_idx_ptr(m, i, j+0), r0);
			_mm256_store_ps(mat_t_idx_ptr(m, i, j+1), r1);
			_mm256_store_ps(mat_t_idx_ptr(m, i, j+2), r2);
			_mm256_store_ps(mat_t_idx_ptr(m, i, j+3), r3);
			_mm256_store_ps(mat_t_idx_ptr(m, i, j+4), r4);
			_mm256_store_ps(mat_t_idx_ptr(m, i, j+5), r5);
			_mm256_store_ps(mat_t_idx_ptr(m, i, j+6), r6);
			_mm256_store_ps(mat_t_idx_ptr(m, i, j+7), r7);
		}
	}
}

#endif
