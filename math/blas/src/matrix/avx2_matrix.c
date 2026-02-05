#ifdef SIMD_AVX2
#include <immintrin.h>

#include "logger.h"

#include "matrix.h"

// Matrix Operation

void mat_cadd(Matrix *mat1, Matrix *mat2, float coef) {
    get_mat_ctx(m1, mat1);
    get_mat_ctx(m2, mat2);

    __m256 m10, m11, m12, m13, m14, m15, m16, m17, m20, m21, m22, m23, m24, m25,
        m26, m27;

    m17 = _mm256_set1_ps(coef);

    for (uint32_t y = 0; y < m1.sy; y += 8) {
        for (uint32_t x = 0; x < m1.sx; x += 8) {
            m20 = _mm256_load_ps(mat_idx_ptr(m2, x + 0, y + 0));
            m21 = _mm256_load_ps(mat_idx_ptr(m2, x + 1, y + 0));
            m22 = _mm256_load_ps(mat_idx_ptr(m2, x + 2, y + 0));
            m23 = _mm256_load_ps(mat_idx_ptr(m2, x + 3, y + 0));
            m24 = _mm256_load_ps(mat_idx_ptr(m2, x + 4, y + 0));
            m25 = _mm256_load_ps(mat_idx_ptr(m2, x + 5, y + 0));
            m26 = _mm256_load_ps(mat_idx_ptr(m2, x + 6, y + 0));
            m27 = _mm256_load_ps(mat_idx_ptr(m2, x + 7, y + 0));

            m10 = _mm256_load_ps(mat_idx_ptr(m1, x + 0, y + 0));
            m11 = _mm256_load_ps(mat_idx_ptr(m1, x + 1, y + 0));
            m12 = _mm256_load_ps(mat_idx_ptr(m1, x + 2, y + 0));
            m13 = _mm256_load_ps(mat_idx_ptr(m1, x + 3, y + 0));
            m14 = _mm256_load_ps(mat_idx_ptr(m1, x + 4, y + 0));
            m15 = _mm256_load_ps(mat_idx_ptr(m1, x + 5, y + 0));
            m16 = _mm256_load_ps(mat_idx_ptr(m1, x + 6, y + 0));

            m10 = _mm256_fmadd_ps(m20, m17, m10);
            m11 = _mm256_fmadd_ps(m21, m17, m11);
            m12 = _mm256_fmadd_ps(m22, m17, m12);
            m13 = _mm256_fmadd_ps(m23, m17, m13);
            m14 = _mm256_fmadd_ps(m24, m17, m14);
            m15 = _mm256_fmadd_ps(m25, m17, m15);
            m16 = _mm256_fmadd_ps(m26, m17, m16);

            _mm256_store_ps(mat_idx_ptr(m1, x + 0, y + 0), m10);
            _mm256_store_ps(mat_idx_ptr(m1, x + 1, y + 0), m11);
            _mm256_store_ps(mat_idx_ptr(m1, x + 2, y + 0), m12);
            _mm256_store_ps(mat_idx_ptr(m1, x + 3, y + 0), m13);
            _mm256_store_ps(mat_idx_ptr(m1, x + 4, y + 0), m14);
            _mm256_store_ps(mat_idx_ptr(m1, x + 5, y + 0), m15);
            _mm256_store_ps(mat_idx_ptr(m1, x + 6, y + 0), m16);

            m10 = _mm256_load_ps(mat_idx_ptr(m1, x + 7, y + 0));
            m10 = _mm256_fmadd_ps(m27, m17, m10);
            _mm256_store_ps(mat_idx_ptr(m1, x + 7, y + 0), m10);
        }
    }
}

void mat_vmul(Matrix *mat, Vector *vec, Vector *res) {
    // Get contexts for the kernel

    get_mat_ctx(mat_ctx, mat);
    get_vec_ctx(vec_ctx, vec);
    get_vec_ctx(res_ctx, res);

    // Compute the multiplication with the kernel

    __m256 v0, v1, v2, v3, v4, v5, v6, v7, m0, m1, m2, m3, m4, m5, m6, m7;
    for (uint32_t x = 0; x < mat_ctx.sy; x += 8) {
        m7 = _mm256_setzero_ps();
        for (uint32_t y = 0; y < mat_ctx.sx; y += 8) {
            v0 = _mm256_set1_ps(vec_idx(vec_ctx, y + 0));
            v1 = _mm256_set1_ps(vec_idx(vec_ctx, y + 1));
            v2 = _mm256_set1_ps(vec_idx(vec_ctx, y + 2));
            v3 = _mm256_set1_ps(vec_idx(vec_ctx, y + 3));
            v4 = _mm256_set1_ps(vec_idx(vec_ctx, y + 4));
            v5 = _mm256_set1_ps(vec_idx(vec_ctx, y + 5));
            v6 = _mm256_set1_ps(vec_idx(vec_ctx, y + 6));
            v7 = _mm256_set1_ps(vec_idx(vec_ctx, y + 7));

            m0 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x + 0, y + 0));
            m1 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x + 0, y + 1));
            m2 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x + 0, y + 2));
            m3 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x + 0, y + 3));
            m4 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x + 0, y + 4));
            m5 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x + 0, y + 5));
            m6 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x + 0, y + 6));

            m7 = _mm256_fmadd_ps(v0, m0, m7);
            m7 = _mm256_fmadd_ps(v1, m1, m7);
            m7 = _mm256_fmadd_ps(v2, m2, m7);
            m7 = _mm256_fmadd_ps(v3, m3, m7);
            m7 = _mm256_fmadd_ps(v4, m4, m7);
            m7 = _mm256_fmadd_ps(v5, m5, m7);
            m7 = _mm256_fmadd_ps(v6, m6, m7);

            m0 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x + 0, y + 7));
            m7 = _mm256_fmadd_ps(v7, m0, m7);
        }

        _mm256_store_ps(vec_idx_ptr(res_ctx, x), m7);
    }
}

void mat_fmva(Matrix *mat, Vector *vec, Vector *res) {
    // Get contexts for the kernel

    get_mat_ctx(mat_ctx, mat);
    get_vec_ctx(vec_ctx, vec);
    get_vec_ctx(res_ctx, res);

    // Compute the multiplication with the kernel

    __m256 v0, v1, v2, v3, v4, v5, v6, v7, m0, m1, m2, m3, m4, m5, m6, m7;
    for (uint32_t x = 0; x < mat_ctx.sy; x += 8) {
        debugm("m7", vec_idx_ptr(res_ctx, x));m7 = _mm256_load_ps(vec_idx_ptr(res_ctx, x));
        for (uint32_t y = 0; y < mat_ctx.sx; y += 8) {
            v0 = _mm256_set1_ps(vec_idx(vec_ctx, y + 0));
            v1 = _mm256_set1_ps(vec_idx(vec_ctx, y + 1));
            v2 = _mm256_set1_ps(vec_idx(vec_ctx, y + 2));
            v3 = _mm256_set1_ps(vec_idx(vec_ctx, y + 3));
            v4 = _mm256_set1_ps(vec_idx(vec_ctx, y + 4));
            v5 = _mm256_set1_ps(vec_idx(vec_ctx, y + 5));
            v6 = _mm256_set1_ps(vec_idx(vec_ctx, y + 6));
            v7 = _mm256_set1_ps(vec_idx(vec_ctx, y + 7));

            m0 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x + 0, y + 0));
            m1 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x + 0, y + 1));
            m2 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x + 0, y + 2));
            m3 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x + 0, y + 3));
            m4 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x + 0, y + 4));
            m5 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x + 0, y + 5));
            m6 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x + 0, y + 6));

            m7 = _mm256_fmadd_ps(v0, m0, m7);
            m7 = _mm256_fmadd_ps(v1, m1, m7);
            m7 = _mm256_fmadd_ps(v2, m2, m7);
            m7 = _mm256_fmadd_ps(v3, m3, m7);
            m7 = _mm256_fmadd_ps(v4, m4, m7);
            m7 = _mm256_fmadd_ps(v5, m5, m7);
            m7 = _mm256_fmadd_ps(v6, m6, m7);

            m0 = _mm256_load_ps(mat_t_idx_ptr(mat_ctx, x + 0, y + 7));
            m7 = _mm256_fmadd_ps(v7, m0, m7);
        }

        _mm256_store_ps(vec_idx_ptr(res_ctx, x), m7);
    }
}

void mat_t_vmul(MatrixTranpose *mat, Vector *vec, Vector *res) {
	debug("Begin Matrix transpose vector multiplication");
    // Get contexts for the kernel

	debug("Retreving Context");
    get_mat_t_ctx(mat_ctx, mat);
    get_vec_ctx(vec_ctx, vec);
    get_vec_ctx(res_ctx, res);
	debug("Retreived");
	debugm("Matrix data", mat_ctx.data);
	debugm("Matrix Tdata", mat_ctx.data_t);
	newline_d();
	debugm("Vector data", vec_ctx.data);
	newline_d();
	debugm("Result data", res_ctx.data);

    // Compute the multiplication with the kernel

	debug("Computing");
    __m256 v0, v1, v2, v3, v4, v5, v6, v7, m0, m1, m2, m3, m4, m5, m6, m7;
    for (uint32_t x = 0; x < mat_ctx.sy; x += 8) {
		debug("Zeroing result 7");
        m7 = _mm256_setzero_ps();
		debug("Zeroed");
        for (uint32_t y = 0; y < mat_ctx.sx; y += 8) {
			debug("Loading vector data");
            v0 = _mm256_set1_ps(vec_idx(vec_ctx, y + 0));
            v1 = _mm256_set1_ps(vec_idx(vec_ctx, y + 1));
            v2 = _mm256_set1_ps(vec_idx(vec_ctx, y + 2));
            v3 = _mm256_set1_ps(vec_idx(vec_ctx, y + 3));
            v4 = _mm256_set1_ps(vec_idx(vec_ctx, y + 4));
            v5 = _mm256_set1_ps(vec_idx(vec_ctx, y + 5));
            v6 = _mm256_set1_ps(vec_idx(vec_ctx, y + 6));
            v7 = _mm256_set1_ps(vec_idx(vec_ctx, y + 7));
			debug("Loaded");

			debug("Loading matrix data");
            debugm("m0", mat_dt_idx_ptr(mat_ctx, x + 0, y + 0));m0 = _mm256_load_ps(mat_dt_idx_ptr(mat_ctx, x + 0, y + 0)); debug("Loaded m0");
            debugm("m1", mat_dt_idx_ptr(mat_ctx, x + 0, y + 1));m1 = _mm256_load_ps(mat_dt_idx_ptr(mat_ctx, x + 0, y + 1)); debug("Loaded m1");
            debugm("m2", mat_dt_idx_ptr(mat_ctx, x + 0, y + 2));m2 = _mm256_load_ps(mat_dt_idx_ptr(mat_ctx, x + 0, y + 2)); debug("Loaded m2");
            debugm("m3", mat_dt_idx_ptr(mat_ctx, x + 0, y + 3));m3 = _mm256_load_ps(mat_dt_idx_ptr(mat_ctx, x + 0, y + 3)); debug("Loaded m3");
            debugm("m4", mat_dt_idx_ptr(mat_ctx, x + 0, y + 4));m4 = _mm256_load_ps(mat_dt_idx_ptr(mat_ctx, x + 0, y + 4)); debug("Loaded m4");
            debugm("m5", mat_dt_idx_ptr(mat_ctx, x + 0, y + 5));m5 = _mm256_load_ps(mat_dt_idx_ptr(mat_ctx, x + 0, y + 5)); debug("Loaded m5");
            debugm("m6", mat_dt_idx_ptr(mat_ctx, x + 0, y + 6));m6 = _mm256_load_ps(mat_dt_idx_ptr(mat_ctx, x + 0, y + 6)); debug("Loaded m6");
			debug("Loaded");

			debug("Computing");
            m7 = _mm256_fmadd_ps(v0, m0, m7);
            m7 = _mm256_fmadd_ps(v1, m1, m7);
            m7 = _mm256_fmadd_ps(v2, m2, m7);
            m7 = _mm256_fmadd_ps(v3, m3, m7);
            m7 = _mm256_fmadd_ps(v4, m4, m7);
            m7 = _mm256_fmadd_ps(v5, m5, m7);
            m7 = _mm256_fmadd_ps(v6, m6, m7);

            debugm("m7", mat_dt_idx_ptr(mat_ctx, x + 0, y + 7));
			m0 = _mm256_load_ps(mat_dt_idx_ptr(mat_ctx, x + 0, y + 7));
			debug("Loaded m7");
            m7 = _mm256_fmadd_ps(v7, m0, m7);
			debug("Computed");
        }

		debug("Storing result");
		debugm("Store ptr", vec_idx_ptr(res_ctx, x));
        _mm256_store_ps(vec_idx_ptr(res_ctx, x), m7);
		debug("Stored");
    }
	debug("Computed");
	debug("End Matrix transpose vector multiplication");
}
#endif
