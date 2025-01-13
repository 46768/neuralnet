# Define the minimum version of CMake
cmake_minimum_required(VERSION 3.15)

# Define the project
project(FCLNN C)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED True)

# Export compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set compiler flags
set(CMAKE_DEBUG_POSTFIX d)
add_library(compiler_flags INTERFACE)
target_compile_features(compiler_flags INTERFACE c_std_23)

set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")

target_compile_options(compiler_flags INTERFACE
	"$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall;-Wextra;-Og;-g>>"
	"$<${msvc_cxx}:$<BUILD_INTERFACE:-W3>>")

# Add the libraries
add_subdirectory("${PROJECT_SOURCE_DIR}/lib") # Common utilities
add_subdirectory("${PROJECT_SOURCE_DIR}/math/linear_alg") # Feed forward network
add_subdirectory("${PROJECT_SOURCE_DIR}/nn") # Neural network
add_subdirectory("${PROJECT_SOURCE_DIR}/nn/ffn") # Feed forward network

# Add main.c as executable
set(ENTRY_POINT "${PROJECT_SOURCE_DIR}/src/main.c")
add_executable(FCLNN "${ENTRY_POINT}")

# Set debug property
set_target_properties(FCLNN PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})

# Link libraries
target_link_libraries(FCLNN PRIVATE ComLib) # Common Utility
target_link_libraries(FCLNN PRIVATE NeuralNetwork) # Feed Forward Network
target_link_libraries(FCLNN PRIVATE FeedForwardNetwork) # Feed Forward Network
target_link_libraries(FCLNN PRIVATE m) # C Math Library
