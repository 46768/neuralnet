#include "python_interface.h"

#include <unistd.h>
#include <sys/wait.h>

#include "logger.h"

void python_spawn(const char* script_path, const char* data_path) {
	if (!data_path) {
		fatal("NULL Data path");
	}
	if (!script_path) {
		fatal("NULL Script name");
	}

	pid_t pid = fork();
	int status;
	if (pid == 0) {
		execlp(PYTHON_CMD, PYTHON_CMD, script_path, data_path, NULL);
	}
	waitpid(pid, &status, 0);
}
