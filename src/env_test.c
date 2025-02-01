#include <stdio.h>

#include "python_interface.h"

int main() {
	python_create_venv(PROJECT_PATH "/requirements.txt");
	printf("Hello world!\n");
	return 0;
}
