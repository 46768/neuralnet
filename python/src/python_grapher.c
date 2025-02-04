#include "python_grapher.h"

#include "python_interface.h"

const char* script_path_grapher = PROJECT_PATH "/python/psrc/grapher.py";
const char* script_path_grapher_pair = PROJECT_PATH "/python/psrc/grapher_pair.py";

void python_graph(const char* csv_filename) {
	python_spawn(script_path_grapher, csv_filename);
}
