# Include all source files
include_directories(
	"${CMAKE_CURRENT_SOURCE_DIR}/src"
)
file(GLOB srcs
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

# Define common libraries
add_library(ComLib STATIC
	${srcs}
)

target_include_directories(ComLib PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/header"
)
