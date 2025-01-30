#include "python_interface.h"

#ifdef SYS_WINDOWS
#	include "process.h"
#else
#	include "unistd.h"
#endif

#include "logger.h"

void python_spawn(const char* script_path, const char* data_path) {
	if (!data_path) {
		fatal("NULL Data path");
	}
	if (!script_path) {
		fatal("NULL Script name");
	}

#ifdef SYS_WINDOWS
	_spawnlp(_P_NOWAIT, PYTHON_CMD, script_path, data_path, NULL);
#else
	pid_t pid = fork();
	if (pid == 0) {
		execlp(PYTHON_CMD, PYTHON_CMD, script_path, data_path, NULL);
	}
#endif
}
