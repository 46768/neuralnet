#include "grapher.h"

#include "python_interface.h"

const char* script_path = PROJECT_PATH "/python/psrc/grapher.py";

void python_graph(const char* csv_path) {
	python_spawn(script_path, csv_path);
}
