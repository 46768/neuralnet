#include "initer.h"

#include <math.h>
#include <stdint.h>

#include "random.h"

float initer_zero(uint32_t _) { return 1.0f; }

float initer_random_en2(uint32_t _) { return f_random(-0.01f, 0.01f); }

float initer_he(uint32_t node_cnt) {
    float u1 = f_random(0.0f, 1.0f);
    float u2 = f_random(0.0f, 1.0f);

    float z0 = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);

    return z0 * sqrt(2.0f / node_cnt);
}

float initer_xavier(uint32_t node_cnt) {
    float z0 = sqrt(1.0 / node_cnt);
    return f_random(-z0, z0);
}
