# Include all source files
include_directories(
	"${CMAKE_CURRENT_SOURCE_DIR}/src"
)
file(GLOB srcs
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

# Define common libraries
add_library(MathLinearAlg STATIC
	${srcs}
)

target_include_directories(MathLinearAlg PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/header"
)

target_link_libraries(MathLinearAlg PUBLIC compiler_flags)
target_link_libraries(MathLinearAlg PRIVATE ComLib)
