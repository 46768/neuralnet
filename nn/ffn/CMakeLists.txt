# Include all source files
include_directories(
	"${CMAKE_CURRENT_SOURCE_DIR}/src"
)
file(GLOB srcs
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

# Define feed forward network libraries
add_library(FeedForwardNetwork STATIC
	${srcs}
)

# Link headers to the library
target_include_directories(FeedForwardNetwork PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/header"
)

# Link other libraries to the library
target_link_libraries(FeedForwardNetwork PUBLIC MathLinearAlg)
target_link_libraries(FeedForwardNetwork PUBLIC NeuralNetwork)
target_link_libraries(FeedForwardNetwork PUBLIC CompilerFlags)
target_link_libraries(FeedForwardNetwork PRIVATE ComLib)
