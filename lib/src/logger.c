#include "logger.h"

#include <libgen.h>
#include <stdio.h>
#include <stdarg.h>

const char* level_header[] = {
	"INFO", // INFO
	"WARN", // WARN
	"ERROR", // ERROR
	"DEBUG", // DEBUG
	"FATAL", // FATAL
};

const char* ansi_level_coding[] = {
	"\x1b[34m", // INFO
	"\x1b[33m", // WARN
	"\x1b[31m", // ERROR
	"\x1b[35m", // DEBUG
	"\x1b[30;41m", // FATAL
};

void _log(
		int level,
		const char* file,
		int line,
		const char* func,
		const char* format,
		...
		) {

	// Log out the header
	fprintf(stderr, "[%s%s\x1b[0m] %s:%s:%d ",
			ansi_level_coding[level], level_header[level],
			basename((char*)file), func, line);

	// Log out the information
	va_list args;
	va_start(args, format);
	vfprintf(stderr, format, args);
	va_end(args);

	fprintf(stderr, "\n");
}

void inline newline() {
	fprintf(stderr, "\n");
}
