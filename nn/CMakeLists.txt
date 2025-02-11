# Include all source files
include_directories(
	"${CMAKE_CURRENT_SOURCE_DIR}/src"
)
file(GLOB srcs
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

# Define neural network libraries
add_library(NeuralNetwork STATIC
	${srcs}
)

# Link headers to the library
target_include_directories(NeuralNetwork PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/header"
)

# Link other libraries to the library
target_link_libraries(NeuralNetwork PUBLIC MathLinearAlg)
target_link_libraries(NeuralNetwork PUBLIC CompilerFlags)
target_link_libraries(NeuralNetwork PUBLIC m)
target_link_libraries(NeuralNetwork PRIVATE ComLib)
