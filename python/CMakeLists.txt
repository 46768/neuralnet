# Include all source files
include_directories(
	"${CMAKE_CURRENT_SOURCE_DIR}/src"
)
file(GLOB srcs
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

add_library(PythonInterface STATIC
	${srcs}
)

# Link headers to the library
target_include_directories(PythonInterface PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/header"
)

target_link_libraries(PythonInterface PUBLIC compiler_flags)
target_link_libraries(PythonInterface PRIVATE ComLib)
