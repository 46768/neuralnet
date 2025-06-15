#include "random.h"

#include <stdlib.h>
#include <time.h>

void init_random() { srand(time(NULL)); }

float f_random(float lower, float upper) {
    float x = (((float)rand() / (float)RAND_MAX) * (upper - lower)) + lower;
    return x;
}
