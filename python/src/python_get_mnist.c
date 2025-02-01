#include "python_get_mnist.h"

#include "python_interface.h"

const char* script_path_get_mnist = PROJECT_PATH "/python/psrc/get_mnist.py";

void python_get_mnist(const char* output_path) {
	python_spawn(script_path_get_mnist, output_path);
}
