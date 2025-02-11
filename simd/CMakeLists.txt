# Include all source files
include_directories(
	"${CMAKE_CURRENT_SOURCE_DIR}/src"
)
file(GLOB srcs
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

# Define common libraries
add_library(SIMDLib STATIC
	${srcs}
)

target_include_directories(SIMDLib PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/header"
)

target_link_libraries(SIMDLib PUBLIC CompilerFlags)
target_link_libraries(SIMDLib PUBLIC ComLib)
