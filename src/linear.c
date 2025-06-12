#include <stdlib.h>

#include "ffn.h"

int main() {
	FFNInitData* initd = ffn_init();
	ffn_add_layer(initd, 784, None, Zero, Zero);
	ffn_add_layer(initd, 16, None, Zero, Zero);
	ffn_add_layer(initd, 16, None, Zero, Zero);
	ffn_set_output(initd, 10);
	ffn_set_cost_fn(initd, MSE);

	FFNModel* model = (FFNModel*)malloc(sizeof(FFNModel));

	ffn_build(initd, model);

	return 0;
}
