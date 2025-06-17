#include <iostream>

#include "random.hpp"

int main() {
    std::cout << "Hello World!\n";
    std::cout << f_random_uniform(-10.0f, 2.0f) << std::endl;
    std::cout << f_random_normal(0.0f, 2.0f) << std::endl;

    return 0;
}
