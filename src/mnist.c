#include <string.h>

#include "logger.h"
#include "random.h"
#include "allocator.h"
#include "file_io.h"

#include "python_interface.h"
#include "python_grapher.h"
#include "python_get_mnist.h"

#include "generator.h"

#include "ffn_init.h"
#include "ffn_fpropagate.h"
#include "ffn_bpropagate.h"
#include "ffn_mempool.h"
#include "ffn_util.h"

int main() {
	debug("init");
	info(PROJECT_PATH);
	python_create_venv(PROJECT_PATH "/requirements.txt");
	python_get_mnist(PROJECT_PATH "/data/mnist");
	init_random();
	// Inputs
	Vector** vecs;
	Vector** targets;
	int REGS_RANGEL = -10;
	int REGS_RANGE = 10;
	//generate_linear_regs(REGS_RANGEL, REGS_RANGE, -4.0f, 5.0f, &vecs, &targets);
	//generate_xor(&REGS_RANGEL, &REGS_RANGE, &vecs, &targets);
	generate_mnist(&REGS_RANGEL, &REGS_RANGE, &vecs, &targets,
			"mnist/train_img.idx",
			"mnist/train_label.idx",
			"mnist/test_img.idx",
			"mnist/test_label.idx"
			);

	return 0;
}
