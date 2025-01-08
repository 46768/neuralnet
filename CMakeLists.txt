# Define the minimum version of CMake
cmake_minimum_required(VERSION 3.15)

# Define the project
project(FCLNN C)
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED True)

# Export compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set compiler flags
add_library(tutorial_compiler_flags INTERFACE)
target_compile_features(tutorial_compiler_flags INTERFACE c_std_23)

set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")

target_compile_options(tutorial_compiler_flags INTERFACE
	"$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall;-Wextra>>"
	"$<${msvc_cxx}:$<BUILD_INTERFACE:-W3>>")

# Add the libraries
add_subdirectory("${PROJECT_SOURCE_DIR}/lib") # Common utilities

# Add main.c as executable
set(ENTRY_POINT "${PROJECT_SOURCE_DIR}/src/main.c")
add_executable(AOC2019 "${ENTRY_POINT}")
