# Define the minimum version of CMake
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

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
include(cmake/FetchCMocka.cmake)

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
target_link_libraries(NNLib
	INTERFACE
	CompilerFlags # Compiler Flags
	Utility # Utilities
)

file(GLOB_RECURSE EntryPoints CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.c")

foreach (EntryPoint ${EntryPoints})
	get_filename_component(TargetExecName ${EntryPoint} NAME_WLE)
	add_executable(${TargetExecName} ${EntryPoint})
	target_link_libraries(${TargetExecName} PRIVATE NNLib)
endforeach()

# Unit tests
file(GLOB_RECURSE UnitTests CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/tests/unit/*.c")

foreach (Test ${UnitTests})
	get_filename_component(TargetTestName ${Test} NAME_WLE)
	add_executable(${TargetTestName} ${Test})
	target_link_libraries(${TargetTestName} PRIVATE cmocka-static)
	set_target_properties(${TargetTestName}
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/unit"
	)
endforeach()
# Unit test runner
file(GENERATE
	OUTPUT "${CMAKE_BINARY_DIR}/UnitTest"
	CONTENT "cd ${CMAKE_BINARY_DIR}/tests/unit; for i in $(ls ${CMAKE_BINARY_DIR}/tests/unit); do echo \"running test $i\"; ./$i; echo ; done"
	FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE
)

# Integration tests
file(GLOB_RECURSE IntegrationTests CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/tests/integration/*.c")

foreach (Test ${IntegrationTests})
	get_filename_component(TargetTestName ${Test} NAME_WLE)
	add_executable(${TargetTestName} ${Test})
	target_link_libraries(${TargetTestName} PRIVATE cmocka-static)
	set_target_properties(${TargetTestName}
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/integration"
	)
endforeach()
# Unit test runner
file(GENERATE
	OUTPUT "${CMAKE_BINARY_DIR}/IntegrationTest"
	CONTENT "cd ${CMAKE_BINARY_DIR}/tests/integration; for i in $(ls ${CMAKE_BINARY_DIR}/tests/integration); do echo \"running test $i\"; ./$i; echo ; done"
	FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE
)
