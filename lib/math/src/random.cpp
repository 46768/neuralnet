#include "random.hpp"

#include <random>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> frandom(0.0f, 1.0f);

float f_random_uniform(float lower, float upper) {
    return (frandom(gen) * (upper - lower)) + lower;
}

float f_random_normal(float mean, float std_dev) {
    std::normal_distribution<float> d{mean, std_dev};

    return d(gen);
}
