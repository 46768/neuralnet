#include "python_test.h"

#include "python_interface.h"

const char* script_path_test = PROJECT_PATH "/python/psrc/test.py";

void python_test() {
	python_spawn(script_path_test, "");
}
