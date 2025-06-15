#include <stdlib.h>

#include "ffn.h"

int main() {
    FFNInitData *initd = ffn_init();
    ffn_add_layer(initd, 768, None, Zero, Zero);
    ffn_add_layer(initd, 32, None, Zero, Zero);
    ffn_add_layer(initd, 32, None, Zero, Zero);
    ffn_set_output(initd, 10);
    ffn_set_cost_fn(initd, MSE);

    FFNModel *model = (FFNModel *)malloc(sizeof(FFNModel));

    ffn_build(initd, model);

	ffn_free(model);
	ffn_free_init(initd);

    return 0;
}
