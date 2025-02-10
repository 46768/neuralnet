# Include all source files
file(GLOB srcs
	"${CMAKE_CURRENT_SOURCE_DIR}/matrix/src/*.c"
	"${CMAKE_CURRENT_SOURCE_DIR}/vector/src/*.c"
)

# Define common libraries
add_library(MathLinearAlg STATIC
	${srcs}
)

target_include_directories(MathLinearAlg PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/matrix/header"
	"${CMAKE_CURRENT_SOURCE_DIR}/vector/header"
)

target_link_libraries(MathLinearAlg PUBLIC compiler_flags)
target_link_libraries(MathLinearAlg PUBLIC ComLib)
target_link_libraries(MathLinearAlg PUBLIC SIMDLib)
