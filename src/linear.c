#include <stdlib.h>

#include "ffn.h"

int main() {
    FFNInitData *initd = ffn_init();
    ffn_add_layer(initd, 2, None, Zero, Zero);
    ffn_set_output(initd, 2);
    ffn_set_cost_fn(initd, MSE);

    FFNModel *model = (FFNModel *)malloc(sizeof(FFNModel));

    ffn_build(initd, model);

    mat_phy_transpose(model->parameter.weight);

    ffn_free(model);
    ffn_free_init(initd);

    return 0;
}
