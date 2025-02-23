# Define the minimum version of CMake
cmake_minimum_required(VERSION 3.15)

# Define the project
# Version numbering: {Major refactors/reworks}.{Major updates}.{Minor updates}.{Patches}
project(MLLib LANGUAGES C VERSION 1.0.0.0)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED True)

# Export compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/Options.cmake) # Define options
include(cmake/SIMDCheck.cmake) # Get SIMD capabilites
include(cmake/CompilerFlags.cmake) # Set compiler flags
include(cmake/PlatformMacros.cmake) # Platform macro
include(cmake/GetTargets.cmake)

# Project root path as macro
target_compile_definitions(CompilerFlags INTERFACE
	PROJECT_PATH="${PROJECT_SOURCE_DIR}"
	PYTHON_CMD="/usr/bin/env python3"
	PYTHON_VENV="${PROJECT_SOURCE_DIR}/data/venv/bin/python3"
)

# Add the libraries
add_subdirectory("${PROJECT_SOURCE_DIR}/utils") # Common utilities
add_subdirectory("${PROJECT_SOURCE_DIR}/math/blas") # Linear algebra
add_subdirectory("${PROJECT_SOURCE_DIR}/simd") # AVX SIMD
add_subdirectory("${PROJECT_SOURCE_DIR}/nn") # Neural network

get_all_targets(TargetList ${CMAKE_CURRENT_LIST_DIR})
foreach (target ${TargetList})
	if (NOT ${target} STREQUAL "CompilerFlags")
		target_link_libraries(${target} PUBLIC CompilerFlags)
	endif()
	if (NOT (${target} STREQUAL "Utility" OR ${target} STREQUAL "CompilerFlags"))
		target_link_libraries(${target} PUBLIC Utility)
	endif()
endforeach()

# Add the commonly used libraries
add_library(NNLib INTERFACE)
target_link_libraries(Utility INTERFACE CompilerFlags)
target_link_libraries(NNLib
	INTERFACE
	CompilerFlags # Compiler Flags
	Utility # Utilities
)

# Add executables
add_executable(Linear "src/linear.c")

# Link libraries
target_link_libraries(Linear PRIVATE NNLib)
