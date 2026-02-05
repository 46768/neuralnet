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


void activation_softmax(Vector* vin, Vector* vout) {
	get_vec_ctx(vi, vin);
	get_vec_ctx(vo, vout);

	// Find largest element of the vector
	float z_max = 0;
	for (uint32_t i = 0; i < vi.size; i++) {
		if (vi.data[i] > z_max) {
			z_max = vi.data[i];
		}
	}

	float exp_sum = 0;
	for (uint32_t i = 0; i < vi.size; i++) {
		float z_exp = exp(vi.data[i] - z_max);
		exp_sum += z_exp;
		vo.data[i] = z_exp;
	}
	for (uint32_t i = 0; i < vi.size; i++) {
		vo.data[i] /= exp_sum;
	}
}
