#include "activation.h"

#include <math.h>
#include <string.h>

void activation_none(Vector *vin, Vector *vout) {
    memcpy(vout->data, vin->data, vin->size * sizeof(float));
}

void activation_none_d(Vector *vin, Vector *vout) {
    get_vec_ctx(vo, vout);
    for (uint32_t i = 0; i < vin->size; i++) {
        vec_idx(vo, i) = 1.0f;
    }
}

static inline float _sigmoid(float x) { return (float)1 / (1 + expf(-x)); }
void activation_sigmoid(Vector *vin, Vector *vout) {
    get_vec_ctx(vi, vin);
    get_vec_ctx(vo, vout);

    for (uint32_t i = 0; i < vo.size; i++) {
        vec_idx(vo, i) = _sigmoid(vec_idx(vi, i));
    }
}
void activation_sigmoid_d(Vector *vin, Vector *vout) {
    get_vec_ctx(vi, vin);
    get_vec_ctx(vo, vout);

    for (uint32_t i = 0; i < vo.size; i++) {
        float sig = _sigmoid(vec_idx(vi, i));
        vec_idx(vo, i) = sig * (1 - sig);
    }
}
