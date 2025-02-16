# Define the minimum version of CMake
cmake_minimum_required(VERSION 3.15)

# Define the project
# Version numbering: {Major refactors/reworks}.{Major updates}.{Minor updates}.{Patches}
project(NeuralNetwork LANGUAGES C VERSION 0.0.1.1)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED True)

include(CheckCSourceRuns)

# Export compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Define options
include(cmake/Options.cmake)
# Get SIMD capabilites
include(cmake/SIMDCheck.cmake)
# Set compiler flags
include(cmake/CompilerFlags.cmake)
# Platform macro
include(cmake/PlatformMacros.cmake)

# Project root path as macro
target_compile_definitions(CompilerFlags INTERFACE
	PROJECT_PATH="${PROJECT_SOURCE_DIR}"
	PYTHON_CMD="/usr/bin/env python3"
	PYTHON_VENV="${PROJECT_SOURCE_DIR}/data/venv/bin/python3"
)

# Add the libraries
add_subdirectory("${PROJECT_SOURCE_DIR}/lib") # Common utilities
add_subdirectory("${PROJECT_SOURCE_DIR}/math/linear_alg") # Linear algebra
add_subdirectory("${PROJECT_SOURCE_DIR}/simd") # AVX SIMD
add_subdirectory("${PROJECT_SOURCE_DIR}/threading") # Multithreading
add_subdirectory("${PROJECT_SOURCE_DIR}/nn") # Neural network
add_subdirectory("${PROJECT_SOURCE_DIR}/nn/ffn") # Feed forward network
add_subdirectory("${PROJECT_SOURCE_DIR}/python") # Python Interface

# Add the commonly used libraries
add_library(FCLNNLib INTERFACE)
target_link_libraries(FCLNNLib INTERFACE CompilerFlags) # Compiler Flags
target_link_libraries(FCLNNLib INTERFACE ComLib) # Common Utility
target_link_libraries(FCLNNLib INTERFACE NeuralNetwork) # Feed Forward Network
target_link_libraries(FCLNNLib INTERFACE FeedForwardNetwork) # Feed Forward Network
target_link_libraries(FCLNNLib INTERFACE PythonInterface) # Python Interface
target_link_libraries(FCLNNLib INTERFACE m) # C Math Library

# Add executables
add_executable(FCLNN "${PROJECT_SOURCE_DIR}/src/linear.c")
add_executable(FCLNNn "${PROJECT_SOURCE_DIR}/src/linear_noise.c")
add_executable(FCLNNt "${PROJECT_SOURCE_DIR}/src/xor.c")
add_executable(MNIST "${PROJECT_SOURCE_DIR}/src/mnist.c")
add_executable(UTEST "${PROJECT_SOURCE_DIR}/src/utest.c")
add_executable(STESTP "${PROJECT_SOURCE_DIR}/src/speed_test_p.c")
add_executable(STESTB "${PROJECT_SOURCE_DIR}/src/speed_test_b.c")

# Link libraries
target_link_libraries(FCLNN PUBLIC FCLNNLib)
target_link_libraries(FCLNNn PUBLIC FCLNNLib)
target_link_libraries(FCLNNt PUBLIC FCLNNLib)
target_link_libraries(MNIST PUBLIC FCLNNLib)
target_link_libraries(UTEST PUBLIC FCLNNLib)
target_link_libraries(STESTP PUBLIC FCLNNLib)
target_link_libraries(STESTB PUBLIC FCLNNLib)
