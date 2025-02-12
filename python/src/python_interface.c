#include "python_interface.h"

#include <unistd.h>
#include <sys/wait.h>
#include <dirent.h>
#include <sys/stat.h>

#include "file_io.h"
#include "allocator.h"
#include "logger.h"

const char shellnix_format[] = "# shell.nix\n"
"let\n"
"# We pin to a specific nixpkgs commit for reproducibility.\n"
"# Last updated: 2024-04-29. Check for new commits at https://status.nixos.org.\n"
"pkgs = import (fetchTarball \"https://github.com/NixOS/nixpkgs/archive/867738f9e61218e552398d2b9e2a4cddb88a5c4f.tar.gz\") {};\n"
"in pkgs.mkShell {\n"
"packages = [\n"
"(pkgs.python312.withPackages (python-pkgs: with python-pkgs; [\n"
"%s\n"
"]))\n"
"];\n"
"}";

void python_spawn(const char* script_path, const char* data_path) {
#ifndef NO_PYTHON
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
#endif
}

void python_create_venv(const char* requirements_path) {
#ifndef NO_PYTHON
	int is_nixos = 0;
	struct stat st;
	if (stat("/nix/store", &st) == 0 && stat("/etc/nixos", &st) == 0) {
		is_nixos = 1;
	}
	if (stat(PROJECT_PATH "/data", &st) != 0) {
		mkdir(PROJECT_PATH "/data", 0777);
	}

	if (stat(PROJECT_PATH "/data/venv", &st) != 0) {
		mkdir(PROJECT_PATH "/data/venv", 0777);
		if (!is_nixos) {
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
		} else {
			mkdir(PROJECT_PATH "/data/venv/bin", 0777);
			pid_t pid = fork();
			int status;
			if (pid == 0) {
				execlp("cp", "cp", PROJECT_PATH "/requirements.txt",
						PROJECT_PATH "/data/requirements.txt", NULL);
			} else if (pid > 0) {
				waitpid(pid, &status, 0);
			} else {
				fatal("Failed to copy requirements.txt");
			}
			FileData* shellnix = get_file_write("venv/bin/shell.nix");
			FileData* pythonsh = get_file_write("venv/bin/python3");
			FileData* requirements = get_file_read("requirements.txt");
			char* requirements_buf = (char*)allocate(requirements->size+1);
			if (fread(requirements_buf, sizeof(char), requirements->size,
						requirements->file_pointer) != requirements->size) {
				fatal("Failed to copy requirements.txt into buffer");
			}
			requirements_buf[requirements->size] = '\0';

			fprintf(shellnix->file_pointer, shellnix_format, requirements_buf);
			fprintf(pythonsh->file_pointer,
					"#!/usr/bin/env bash\n"
					"cmd=\"python3 ${@}\"\n"
					"nix-shell %s/data/venv/bin/shell.nix --run \"${cmd}\"",
					PROJECT_PATH);

			close_file(shellnix);
			close_file(pythonsh);
			close_file(requirements);

			chmod(PROJECT_PATH "/data/venv/bin/python3", 0777);
		}

		info("Created python venv");
	}
#endif
}
