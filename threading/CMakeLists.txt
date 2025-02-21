# Include all source files
include_directories(
	"${CMAKE_CURRENT_SOURCE_DIR}/src"
)
file(GLOB srcs
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

# Define common libraries
add_library(ThreadingLib STATIC
	${srcs}
)

target_include_directories(ThreadingLib PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/header"
)

target_link_libraries(ThreadingLib PUBLIC CompilerFlags)
target_link_libraries(ThreadingLib PUBLIC ComLib)
