#include "python_interface.h"

#include <unistd.h>
#include <sys/wait.h>
#include <dirent.h>
#include <sys/stat.h>

#include "logger.h"

void python_spawn(const char* script_path, const char* data_path) {
	struct stat st;
	if (!data_path) {
		fatal("NULL Data path");
	}
	if (!script_path) {
		fatal("NULL Script name");
	}
	if (stat(PROJECT_PATH "/data/venv", &st) != 0) {
		fatal("Python venv not found, call `python_create_venv` first");
	}

	pid_t pid = fork();
	int status;
	if (pid == 0) {
		execlp(PYTHON_VENV, PYTHON_VENV, script_path, data_path, NULL);
	} else if (pid > 0) {
		waitpid(pid, &status, 0);
	} else {
		fatal("Fork failed for python_venv %s", script_path);
	}
}

void python_create_venv(const char* requirements_path) {
	struct stat st;
	if (stat(PROJECT_PATH "/data", &st) != 0) {
		mkdir(PROJECT_PATH "/data", 0777);
	}

	if (stat(PROJECT_PATH "/data/venv", &st) != 0) {
		mkdir(PROJECT_PATH "/data/venv", 0777);
		pid_t pid1 = fork();
		int status;
		if (pid1 == 0) {
			execlp(PYTHON_CMD, PYTHON_CMD, "-m", "venv", PROJECT_PATH "/data/venv", NULL);
		} else if (pid1 > 0) {
			waitpid(pid1, &status, 0);
		} else {
			fatal("Fork failed for python3 venv");
		}
		info("Created venv");
		pid_t pid2 = fork();
		if (pid2 == 0) {
			execlp(PROJECT_PATH "/data/venv/bin/pip", PROJECT_PATH "/data/venv/bin/pip",
					"install", "-r", requirements_path, NULL);
		} else if (pid2 > 0) {
			waitpid(pid2, &status, 0);
		} else {
			fatal("Fork failed for pip install");
		}
		info("Installed packages");
	}
	info("Created python venv");
}
