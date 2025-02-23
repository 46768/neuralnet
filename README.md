# C's C Neural Network Library
A made-from-scratch neural network and machine learning library **not** designed to be used in production, but
more for experimentations

---

## Building
### Required tools
- CMake 3.15+
- C compiler (GCC/Clang)
- Python 3
- Linux environment
```sh
# Clone the project
$ git clone https://github.com/46768/neuralnet.git
$ cd /path/to/project

# Create target directory
$ mkdir target
$ cd target
$ cmake ..

# Cmake options
# -DPROFILING: Profiling with gprof
# -DUSE_SCALAR: Use scalar rather than SIMD
# -DPROD: use -O2 than -Og
# -DNO_PYTHON: turn off python integration, disable graphing and dataset downloading
# -DNO_BOUND_CHECK: turn off vector/matrix bound checking
# -DNO_STATE_CHECK: turn off model's immutability checking
# -DNEED_SPEED: turn on NO_BOUND_CHECK, NO_STATE_CHECK, and use -O3 than -Og

# Build the project
$ make
```
