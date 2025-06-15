#include "ffn.h"

#include <stdlib.h>

#include "matrix.h"
#include "vector.h"

#include "activation.h"
#include "cost.h"
#include "initer.h"

#ifdef SIMD_AVX
#include "avx.h"
#define data_alloc(s) avx_allocate(s)
#define data_pad(s) ((s + 63) & ~63) - s
#else
#define data_alloc(s) malloc(s)
#define data_pad(s) 0
#endif
#define FLOAT_S sizeof(float)

FFNInitData *ffn_init() {
    FFNInitData *initd = (FFNInitData *)malloc(sizeof(FFNInitData));
    initd->layer_cnt = 0;
    initd->layer_cap = 1;
    initd->cost_fn = MSE;
    initd->layer_data = (FFNLayerData *)malloc(sizeof(FFNLayerData));

    return initd;
}

void ffn_add_layer(FFNInitData *initd, uint32_t l_size,
                   ActivationEnum activation_fn, IniterEnum weight_initer,
                   IniterEnum bias_initer) {
    if (initd->layer_cnt >= initd->layer_cap) {
        initd->layer_cap *= 2;
        initd->layer_data = (FFNLayerData *)realloc(
            initd->layer_data, initd->layer_cap * sizeof(FFNLayerData));
    }

    uint32_t layer_cnt = initd->layer_cnt++;
    FFNLayerData *layer_data = initd->layer_data;

    layer_data[layer_cnt].activation_fn = activation_fn;
    layer_data[layer_cnt].w_initier = weight_initer;
    layer_data[layer_cnt].b_initier = bias_initer;
    layer_data[layer_cnt].size = l_size;
}

void ffn_set_output(FFNInitData *initd, uint32_t o_size) {
    ffn_add_layer(initd, o_size, None, Zero, Zero);
}

void ffn_set_cost_fn(FFNInitData *initd, CostEnum cost_fn) {
    initd->cost_fn = cost_fn;
}

void _ffn_init_parameter(FFNInitData *initd, FFNModel *model) {
    uint32_t l_cnt = initd->layer_cnt;
    FFNLayerData *layer = initd->layer_data;

    model->parameter.cost = cost_resolve(initd->cost_fn);
    model->parameter.cost_d = cost_d_resolve(initd->cost_fn);

    uint64_t w_mdata_size = (l_cnt - 1) * sizeof(MatrixTranpose);
    uint64_t b_mdata_size = (l_cnt - 1) * sizeof(Vector);
    uint64_t a_fn_mdata_size = (l_cnt - 1) * sizeof(ActivationFn);
    uint64_t a_fnd_mdata_size = (l_cnt - 1) * sizeof(ActivationFnD);

    uint64_t mdata_size =
        w_mdata_size + b_mdata_size + a_fn_mdata_size + a_fnd_mdata_size;

    uint32_t padding = data_pad(mdata_size);

    uint64_t data_size = 0;
    for (uint32_t l = 1; l < l_cnt; l++) {
        data_size += calc_vec_size(layer[l].size);
        data_size += calc_mat_size(layer[l - 1].size, layer[l].size) * 2;
    }
    data_size *= FLOAT_S;

    void *dptr = data_alloc(mdata_size + padding + data_size);

    Vector *b_ptr = (Vector *)dptr;
    MatrixTranpose *w_ptr = (MatrixTranpose *)(b_ptr + (l_cnt - 1));
    ActivationFn *a_fn_ptr = (ActivationFn *)(w_ptr + (l_cnt - 1));
    ActivationFnD *a_fnd_ptr = (ActivationFnD *)(a_fn_ptr + (l_cnt - 1));

    float *d_ptr = (float *)((((char *)(a_fnd_ptr + (l_cnt - 1))) + padding));

    uint64_t d_offset = 0;

    for (uint32_t l = 1; l < l_cnt; l++) {
        uint32_t l_size = layer[l].size;
        uint32_t v_size = calc_vec_size(l_size);

        vec_init(l_size, d_ptr + d_offset, b_ptr + (l - 1));

        d_offset += v_size;
    }

    for (uint32_t l = 1; l < l_cnt; l++) {
        uint32_t l_size = layer[l - 1].size;
        uint32_t l1_size = layer[l].size;
        uint64_t m_size = calc_mat_size(l_size, l1_size);

        mat_t_init(l_size, l1_size, d_ptr + d_offset, d_ptr + d_offset + m_size,
                   w_ptr);

        d_offset += m_size * 2;
    }

    model->parameter.bias = b_ptr;
    model->parameter.weight = w_ptr;

    model->parameter.activation = a_fn_ptr;
    model->parameter.activation_d = a_fnd_ptr;

    model->parameter.data = dptr;
}

void _ffn_init_propagation(FFNInitData *initd, FFNModel *model) {
    uint32_t l_cnt = initd->layer_cnt;
    FFNLayerData *layer = initd->layer_data;

    uint64_t preact_mdata_size = l_cnt * sizeof(Vector);
    uint64_t act_mdata_size = l_cnt * sizeof(Vector);

    uint64_t mdata_size = preact_mdata_size + act_mdata_size;

    uint32_t padding = data_pad(mdata_size);

    uint64_t data_size = 0;
    for (uint32_t l = 0; l < l_cnt; l++) {
        data_size += calc_vec_size(layer[l].size) * 2;
    }
    data_size *= FLOAT_S;

    void *dptr = data_alloc(mdata_size + padding + data_size);

    Vector *preact_ptr = (Vector *)dptr;
    Vector *act_ptr = preact_ptr + l_cnt;

    float *d_ptr = (float *)(((char *)(act_ptr + l_cnt)) + padding);

    uint64_t d_offset = 0;

    for (uint32_t l = 0; l < l_cnt; l++) {
        uint32_t l_size = layer[l].size;

        vec_init(l_size, d_ptr + d_offset, preact_ptr + l);

        d_offset += calc_vec_size(l_size);
    }

    for (uint32_t l = 0; l < l_cnt; l++) {
        uint32_t l_size = layer[l].size;

        vec_init(l_size, d_ptr + d_offset, act_ptr + l);

        d_offset += calc_vec_size(l_size);
    }

    model->propagation.preactivation = preact_ptr;
    model->propagation.activation = act_ptr;
    model->propagation.data = dptr;
}

void _ffn_init_gradient(FFNInitData *initd, FFNModel *model) {
    uint32_t l_cnt = initd->layer_cnt;
    FFNLayerData *layer = initd->layer_data;

    uint64_t b_mdata_size = (l_cnt - 1) * sizeof(Vector);
    uint64_t w_mdata_size = (l_cnt - 1) * sizeof(Matrix);
    uint64_t c_mdata_size = sizeof(Vector);

    uint64_t mdata_size = b_mdata_size + w_mdata_size + c_mdata_size;

    uint32_t padding = data_pad(mdata_size);

    uint64_t data_size = 0;
    for (uint32_t l = 1; l < l_cnt; l++) {
        data_size += calc_vec_size(layer[l].size);
        data_size += calc_mat_size(layer[l - 1].size, layer[l].size);
    }
    data_size += calc_vec_size(layer[l_cnt - 1].size);
    data_size *= FLOAT_S;

    void *dptr = data_alloc(mdata_size + padding + data_size);

    Vector *b_ptr = (Vector *)dptr;
    Vector *c_ptr = b_ptr + (l_cnt - 1);
    Matrix *w_ptr = (Matrix *)(c_ptr + 1);

    float *d_ptr = (float *)((((char *)(w_ptr + (l_cnt - 1))) + padding));

    uint64_t d_offset = 0;

    for (uint32_t l = 1; l < l_cnt; l++) {
        uint32_t l_size = layer[l].size;

        vec_init(l_size, d_ptr + d_offset, b_ptr + (l - 1));

        d_offset += calc_vec_size(l_size);
    }

    vec_init(layer[l_cnt - 1].size, d_ptr + d_offset, c_ptr);
    d_offset += calc_vec_size(layer[l_cnt - 1].size);

    for (uint32_t l = 1; l < l_cnt; l++) {
        uint32_t l_size = layer[l - 1].size;
        uint32_t l1_size = layer[l].size;

        mat_init(l_size, l1_size, d_ptr + d_offset, w_ptr + (l - 1));

        d_offset += calc_mat_size(l_size, l1_size);
    }

    model->gradient.bias = b_ptr;
    model->gradient.weight = w_ptr;
    model->gradient.cost = c_ptr;
    model->gradient.data = dptr;
}

void _ffn_init_intermediate(FFNInitData *initd, FFNModel *model) {
    uint32_t l_cnt = initd->layer_cnt;
    FFNLayerData *layer = initd->layer_data;

    uint64_t l_deriv_mdata_size = l_cnt * sizeof(Vector);
    uint64_t err_coef_mdata_size = (l_cnt - 1) * sizeof(Vector);

    uint64_t mdata_size = l_deriv_mdata_size + err_coef_mdata_size;

    uint32_t padding = data_pad(mdata_size);

    uint64_t data_size = 0;
    for (uint32_t l = 0; l < (l_cnt - 1); l++) {
        uint32_t l_size = layer[l].size;

        data_size += calc_vec_size(l_size) * 2;
    }
    data_size += calc_vec_size(layer[l_cnt - 1].size);
    data_size *= FLOAT_S;

    void *dptr = data_alloc(mdata_size + padding + data_size);

    Vector *ld_ptr = (Vector *)dptr;
    Vector *ec_ptr = ld_ptr + l_cnt;

    float *d_ptr = (float *)(((char *)(ec_ptr + l_cnt)) + padding);

    uint64_t d_offset = 0;

    for (uint32_t l = 0; l < l_cnt; l++) {
        uint32_t l_size = layer[l].size;

        vec_init(l_size, d_ptr + d_offset, ld_ptr + l);

        d_offset += calc_vec_size(l_size);
    }

    for (uint32_t l = 0; l < (l_cnt - 1); l++) {
        uint32_t l_size = layer[l].size;

        vec_init(l_size, d_ptr + d_offset, ld_ptr + l);

        d_offset += calc_vec_size(l_size);
    }

    model->intermediate.layer_deriv = ld_ptr;
    model->intermediate.err_coef = ec_ptr;
    model->intermediate.data = dptr;
}

void ffn_build(FFNInitData *initd, FFNModel *model) {
    model->layer_cnt = initd->layer_cnt;

    _ffn_init_parameter(initd, model);
    _ffn_init_propagation(initd, model);
    _ffn_init_gradient(initd, model);
    _ffn_init_intermediate(initd, model);
}

void ffn_free(FFNModel *model) {
    free(model->parameter.data);
    free(model->propagation.data);
    free(model->gradient.data);
    free(model->intermediate.data);
    free(model);
}
