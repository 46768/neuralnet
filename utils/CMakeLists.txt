set(LIB_NAME Utility)

add_library(${LIB_NAME} STATIC)

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
target_sources(${LIB_NAME} PRIVATE ${srcs})

target_include_directories(${LIB_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/header")
