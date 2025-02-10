# C's C Neural Network Library
A made-from-scratch neural network and machine learning library **not** designed to be used in production, but
more for experimentations

---

## Currently Implemented Networks
- MLP/FFN/FCLNN (Multilayer perceptron)/(Feed forward network)/(Fully connected layer neural network)
: Neural network where each node/neurons are connected to every node/neurons in the next layer

---

## Planned Networks
- CNN (Convolutional neural network)
: Image processing network using convolution, and pooling with a FFN at the end
- RNN (Recurrent neural network)
: Neural network where current iteration's activation is used in the next iterations
- LSTMNN (Long short term memory neural network)
: RNN but fixes exploding/vanishing activations
- Transformer

---

## Technology Used
- AVX SIMD
: Vectorization to speed up processing time
- CMake build system
: Build system to compile everything together
- Matplotlib
: Visualization tool for analyzing network loss, or other data

---

## Building
### Requird tools
- Building requires at least CMake 3.15
- C compiler (GCC/Clang)
- Python 3
```sh
# Clone the project
$ git clone https://github.com/46768/neuralnet.git
$ cd /path/to/project

# Create target directory
$ mkdir target
$ cd target
$ cmake ..

# Build the project
$ make

# Running the project
# Linear regression
$ ./FCLNN
# Noised linear regression
$ ./FCLNNn
# XOR dataset
$ ./FCLNNt
# MNIST dataset
$ ./MNIST
# MNIST dataset (debugging)
$ ./MNISTt
# System testing
$ ./UTEST
```

---

# Codebase Organization
```
root
+-- lib - commonly used utility functions
+-- math - custom math library
|   +-- linear_alg - linear algebra functions
|       +-- matrix - matrix implementation
|       +-- vector - linear algebra vector implementation
+-- nn - Neural network library
|   +-- ffn - Feed forward network implementation
|   +-- activation.c - Activation functions collection
|   +-- cost.c - Cost functions collection
|   +-- generator.c - Dataset generators
|   +-- initer.c - Parameter initializer functions collection
+-- python - C-Python interface library
|   +-- psrc - Python source code files
|   |   +-- get_mnist.py - MNIST Dataset downloader
|   |   +-- grapher.py - General line grapher
|   +-- python_get_mnist.c - C interface for get_mnist.py
|   +-- python_grapher.c - C interface for grapher.py
|   +-- python_interface.c - General C interface for initializing venv, and running python script
+-- simd - SIMD library
|   +-- avx.c - AVX wrapper for float pointer
|   +-- avxmm256.c - AVX wrapper for __m256, lower level API than avx.c
+-- requirements.txt - Required python packages used by python_interface.c
```
