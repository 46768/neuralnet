# Define the minimum version of CMake
cmake_minimum_required(VERSION 3.15)

# Define the project
project(FCLNN C)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED True)

# Include compiler flag checker
include(CheckCSourceRuns)

# Export compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler flags library
set(CMAKE_DEBUG_POSTFIX d)
add_library(compiler_flags INTERFACE)
target_compile_features(compiler_flags INTERFACE c_std_17)

# Define options
option(PROFILING "Profile the program" OFF)

# Get compiler type
set(gcc_like_c "$<COMPILE_LANG_AND_ID:C,ARMClang,AppleClang,Clang,GNU,LCC>")
set(msvc_c "$<COMPILE_LANG_AND_ID:C,MSVC>")

# Get AVX2 availability
set(CMAKE_REQUIRED_FLAGS "-mavx2")
check_c_source_runs("
#include <immintrin.h>
int main() {
__m256i v = _mm256_setzero_si256();
return 0;
}" haveAVX2
)
# Get AVX availability
set(CMAKE_REQUIRED_FLAGS "-mavx")
check_c_source_runs("
#include <immintrin.h>
int main() {
__m256 v = _mm256_set1_ps(1.0f);
return 0;
}" haveAVX
)
set(CMAKE_REQUIRED_FLAGS "")

# Set compiler flags
target_compile_options(compiler_flags INTERFACE
	"$<${gcc_like_c}:$<BUILD_INTERFACE:-Wall;-Wextra;-Og;-g>>"
	"$<${msvc_c}:$<BUILD_INTERFACE:/W3>>")

if (PROFILING)
	target_compile_options(compiler_flags INTERFACE
		"$<${gcc_like_c}:$<BUILD_INTERFACE:-pg>>")
	target_link_options(compiler_flags INTERFACE
		"$<${gcc_like_c}:$<BUILD_INTERFACE:-g;-pg>>")
endif()
if (haveAVX2 AND haveAVX)
	target_compile_options(compiler_flags INTERFACE
		"$<${gcc_like_c}:$<BUILD_INTERFACE:-mavx2;-mfma>>")
	target_compile_definitions(compiler_flags INTERFACE "SIMD_AVX")
	target_compile_definitions(compiler_flags INTERFACE "SIMD_AVX2")
elseif (haveAVX)
	target_compile_options(compiler_flags INTERFACE
		"$<${gcc_like_c}:$<BUILD_INTERFACE:-mavx>>")
	target_compile_definitions(compiler_flags INTERFACE "SIMD_AVX")
else()
	target_compile_definitions(compiler_flags INTERFACE "SIMD_NONE")
endif()

# Platform macro
if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
	target_compile_definitions(compiler_flags INTERFACE SYS_LINUX)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
	target_compile_definitions(compiler_flags INTERFACE SYS_WINDOWS)
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
	target_compile_definitions(compiler_flags INTERFACE SYS_MACOS)
else()
	target_compile_definitions(compiler_flags INTERFACE SYS_UNKNOWN)
endif()

# Project root path as macro
target_compile_definitions(compiler_flags INTERFACE
	PROJECT_PATH="${PROJECT_SOURCE_DIR}"
	PYTHON_CMD="python3"
	PYTHON_VENV="${PROJECT_SOURCE_DIR}/data/venv/bin/python3"
)

# Add the libraries
add_subdirectory("${PROJECT_SOURCE_DIR}/lib") # Common utilities
add_subdirectory("${PROJECT_SOURCE_DIR}/math/linear_alg") # Feed forward network
add_subdirectory("${PROJECT_SOURCE_DIR}/simd/avx") # AVX SIMD
add_subdirectory("${PROJECT_SOURCE_DIR}/nn") # Neural network
add_subdirectory("${PROJECT_SOURCE_DIR}/nn/ffn") # Feed forward network
add_subdirectory("${PROJECT_SOURCE_DIR}/python") # Python Interface

# Add the commonly used libraries
add_library(FCLNNLib INTERFACE)
target_link_libraries(FCLNNLib INTERFACE compiler_flags) # Compiler Flags
target_link_libraries(FCLNNLib INTERFACE ComLib) # Common Utility
target_link_libraries(FCLNNLib INTERFACE NeuralNetwork) # Feed Forward Network
target_link_libraries(FCLNNLib INTERFACE FeedForwardNetwork) # Feed Forward Network
target_link_libraries(FCLNNLib INTERFACE PythonInterface) # Python Interface
target_link_libraries(FCLNNLib INTERFACE m) # C Math Library

# Add executables
add_executable(FCLNN "${PROJECT_SOURCE_DIR}/src/main.c")
add_executable(FCLNNt "${PROJECT_SOURCE_DIR}/src/test.c")
add_executable(MNIST "${PROJECT_SOURCE_DIR}/src/mnist.c")
add_executable(UTEST "${PROJECT_SOURCE_DIR}/src/utest.c")

# Set debug property
set_target_properties(FCLNN PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
set_target_properties(FCLNNt PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
set_target_properties(MNIST PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
set_target_properties(UTEST PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})

# Link libraries
target_link_libraries(FCLNN PUBLIC FCLNNLib)
target_link_libraries(FCLNNt PUBLIC FCLNNLib)
target_link_libraries(MNIST PUBLIC FCLNNLib)
target_link_libraries(UTEST PUBLIC FCLNNLib)
