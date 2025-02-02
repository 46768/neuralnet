#include "endianness.h"

#include <stdint.h>

int32_t swap_endian_int(int32_t x) {
	return
		((x << 24) & 0xFF000000) |
		((x << 8) & 0x00FF0000) |
		((x >> 8) & 0x0000FF00) |
		((x >> 24) & 0x000000FF);
}
