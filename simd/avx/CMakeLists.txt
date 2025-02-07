# Include all source files
include_directories(
	"${CMAKE_CURRENT_SOURCE_DIR}/src"
)
file(GLOB srcs
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

# Define common libraries
add_library(AVXLib STATIC
	${srcs}
)

target_include_directories(AVXLib PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/header"
)

target_link_libraries(AVXLib PUBLIC compiler_flags)
target_link_libraries(AVXLib PUBLIC ComLib)
