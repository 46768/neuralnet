#include "initer.h"

#include <math.h>

#include "random.h"
#include "logger.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

Initer resolve_initer(IniterEnum init_type) {
	switch (init_type) {
		case Zero:
			return zero_init;
		case He:
			return he_init;
		case RandomEN2:
			return random_en2_init;
		default:
			fatal("Unknown Initializer");
			return NULL;
	}
}

float zero_init(size_t _) {
	return 0.0f;
}

float he_init(size_t node_cnt) {
	float u1 = f_random(0.0f, 1.0f);
	float u2 = f_random(0.0f, 1.0f);

	float z0 = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);

	return z0 * sqrt(2.0f / node_cnt);
}

float random_en2_init(size_t _) {
	return f_random(-0.01f, 0.01f);
}
