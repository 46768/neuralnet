set(LIB_NAME FFNLib)

add_library(${LIB_NAME} STATIC)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
file(GLOB_RECURSE srcscpp CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
target_sources(${LIB_NAME} PRIVATE ${srcs} ${srcscpp})

target_include_directories(${LIB_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/header")

target_link_libraries(${LIB_NAME}
	PRIVATE MathLib
	PRIVATE BLAS
	PRIVATE NeuralNetLib
)
