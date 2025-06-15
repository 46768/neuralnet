#ifndef NN_FFN_H
#define NN_FFN_H

#include <stdint.h>

#include "matrix.h"
#include "vector.h"

#include "activation.h"
#include "cost.h"
#include "initer.h"

typedef struct {
    uint32_t size;
    ActivationEnum activation_fn;
    IniterEnum w_initier;
    IniterEnum b_initier;
} FFNLayerData;

typedef struct {
    uint32_t layer_cnt;
    uint32_t layer_cap;
    FFNLayerData *layer_data;
    CostEnum cost_fn;
} FFNInitData;

typedef struct {
    uint32_t layer_cnt;

    struct {
        MatrixTranpose *weight;
        Vector *bias;
        ActivationFn *activation;
        ActivationFnD *activation_d;
        CostFn cost;
        CostFnD cost_d;

        void *data;
    } parameter;

    struct {
        Vector *preactivation;
        Vector *activation;

        void *data;
    } propagation;

    struct {
        Matrix *weight;
        Vector *bias;
        Vector *cost;

        void *data;
    } gradient;

    struct {
        Vector *layer_deriv;
        Vector *err_coef;

        void *data;
    } intermediate;
} FFNModel;

FFNInitData *ffn_init();
void ffn_add_layer(FFNInitData *, uint32_t, ActivationEnum, IniterEnum,
                   IniterEnum);
void ffn_set_output(FFNInitData *, uint32_t);
void ffn_set_cost_fn(FFNInitData *, CostEnum);

void ffn_build(FFNInitData *, FFNModel *);
void ffn_free(FFNModel *);

#endif
