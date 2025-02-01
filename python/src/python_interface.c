#include "python_interface.h"

#include <unistd.h>
#include <sys/wait.h>
#include <dirent.h>
#include <sys/stat.h>

#include "logger.h"

void python_spawn(const char* script_path, const char* data_path) {
	if (!data_path) {
		fatal("NULL Data path");
	}
	if (!script_path) {
		fatal("NULL Script name");
	}
	DIR* venv = opendir(PROJECT_PATH "/data/venv");
	if (venv == NULL) {
		fatal("Python venv not found, call `python_create_venv` first");
	}
	closedir(venv);

	pid_t pid = fork();
	int status;
	if (pid == 0) {
		execlp(PYTHON_CMD, PYTHON_CMD, script_path, data_path, NULL);
	}
	waitpid(pid, &status, 0);
}

void python_create_venv(const char* requirements_path) {
	DIR* venv = opendir(PROJECT_PATH "/data/venv");
	mkdir(PROJECT_PATH "/data", 0777);
	if (venv == NULL) {
		mkdir(PROJECT_PATH "/data/venv", 0777);
		pid_t pid1 = fork(), pid2;
		int status;
		execlp(PYTHON_CMD, PYTHON_CMD, "-m", "venv", PROJECT_PATH "/data/venv", NULL);
		waitpid(pid1, &status, 0);
		info("Created venv");
		pid2 = fork();
		execlp(PROJECT_PATH "/data/venv/bin/pip", PROJECT_PATH "/data/venv/bin/pip",
				"install", "-r", requirements_path, NULL);
		waitpid(pid2, &status, 0);
		info("Installed packages");
	}
	info("Created python venv");
}
