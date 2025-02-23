set(LIB_NAME NeuralNetwork)

add_library(${LIB_NAME} STATIC)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
target_sources(${LIB_NAME} PRIVATE ${srcs})

target_include_directories(${LIB_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/header")

target_link_libraries(${LIB_NAME}
	PUBLIC NNLib
	PRIVATE BLAS
)
