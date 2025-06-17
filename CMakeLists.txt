# Define the minimum version of CMake
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

# Define the project
# Version numbering: {Major refactors/reworks}.{Major updates}.{Minor updates}.{Patches}
project(MLLib LANGUAGES C CXX VERSION 1.0.0.0)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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
)

set(PROJECT_LIB_DIR "${PROJECT_SOURCE_DIR}/lib")

# Add the libraries
add_subdirectory("${PROJECT_LIB_DIR}/math") # General math
add_subdirectory("${PROJECT_LIB_DIR}/math/blas") # Linear algebra
add_subdirectory("${PROJECT_LIB_DIR}/neuralnet") # Neural network
add_subdirectory("${PROJECT_LIB_DIR}/neuralnet/ffn") # Feedforward network

get_all_targets(TargetList ${CMAKE_CURRENT_LIST_DIR})
foreach (target ${TargetList})
	if (NOT ${target} STREQUAL "CompilerFlags")
		target_link_libraries(${target} PUBLIC CompilerFlags)
	endif()
endforeach()


# Add the commonly used libraries
add_library(NNLib INTERFACE)
target_link_libraries(NNLib
	INTERFACE
	CompilerFlags # Compiler Flags
	MathLib
	BLAS
	FFNLib
)

# Generate Executables
file(GLOB_RECURSE EntryPoints CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.c")
file(GLOB_RECURSE EntryPointsCXX CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")

foreach (EntryPoint ${EntryPoints})
	get_filename_component(TargetExecName ${EntryPoint} NAME_WLE)
	add_executable(${TargetExecName} ${EntryPoint})
	target_link_libraries(${TargetExecName} PRIVATE NNLib)
endforeach()

foreach (EntryPoint ${EntryPointsCXX})
	get_filename_component(TargetExecName ${EntryPoint} NAME_WLE)
	add_executable(${TargetExecName} ${EntryPoint})
	target_link_libraries(${TargetExecName} PRIVATE NNLib)
endforeach()
