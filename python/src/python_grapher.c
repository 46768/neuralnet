#include "python_grapher.h"

#include "python_interface.h"

const char* script_path_grapher = PROJECT_PATH "/python/psrc/grapher.py";
const char* script_path_gradient_plotter = PROJECT_PATH "/python/psrc/gradient_plotter.py";

void python_graph(const char* csv_filename) {
	python_spawn(script_path_grapher, csv_filename);
}

void python_plot_gradient(const char* csv_filename) {
	python_spawn(script_path_gradient_plotter, csv_filename);
}
