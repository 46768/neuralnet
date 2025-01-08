// Uncomment to enable debugging
#define DEBUG_MODE
// Uncomment to enable verbose logging
//#define VERBOSE_MODE

#ifndef COM_LOGGER_H
#define COM_LOGGER_H

// Level - type
// 0 - INFO
// 1 - WARN
// 2 - ERROR
// 3 - DEBUG
// 4 - FATAL

void _log(int, const char*, int, const char*, const char*, ...);
void newline();

#define info(format, ...) _log(0, __FILE__, __LINE__, __func__, format, ##__VA_ARGS__)
#define warn(format, ...) _log(1, __FILE__, __LINE__, __func__, format, ##__VA_ARGS__)
#define error(format, ...) _log(2, __FILE__, __LINE__, __func__, format, ##__VA_ARGS__)
#define fatal(format, ...) _log(4, __FILE__, __LINE__, __func__, format, ##__VA_ARGS__)

#ifdef DEBUG_MODE
#	define debug(format, ...) _log(3, __FILE__, __LINE__, __func__, format, ##__VA_ARGS__)
#else
#	define debug(format, ...)
#endif

#ifdef VERBOSE_MODE
#	define info_v(format, ...) log(format, ##__VA_ARGS__)
#	define warn_v(format, ...) warn(format, ##__VA_ARGS__)
#	define error_v(format, ...) error(format, ##__VA_ARGS__)
#	define debug_v(format, ...) debug(format, ##__VA_ARGS__)
#	define fatal_v(format, ...) fatal(format, ##__VA_ARGS__)
#else
#	define info_v(format, ...)
#	define warn_v(format, ...)
#	define error_v(format, ...)
#	define debug_v(format, ...)
#	define fatal_v(format, ...)
#endif

#endif
