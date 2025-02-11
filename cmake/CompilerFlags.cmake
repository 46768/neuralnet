add_library(CompilerFlags INTERFACE)
target_compile_features(CompilerFlags INTERFACE c_std_17)

# Compiler flags
set(CompilerFlagss "-Wall;-Wextra;-Og;-g")

# Set compiler flags
target_compile_options(CompilerFlags INTERFACE
	"$<BUILD_INTERFACE:${CompilerFlagss}>")

if (PROFILING)
	target_compile_options(CompilerFlags INTERFACE
		"$<BUILD_INTERFACE:-pg>")
	target_link_options(CompilerFlags INTERFACE
		"$<BUILD_INTERFACE:-g;-pg>")
endif()

if (USE_SCALAR)
	target_compile_definitions(CompilerFlags INTERFACE "SIMD_NONE")
elseif (haveAVX2 AND haveAVX)
	target_compile_options(CompilerFlags INTERFACE
		"$<BUILD_INTERFACE:-mavx2;-mfma>")
	target_compile_definitions(CompilerFlags INTERFACE "SIMD_AVX")
	target_compile_definitions(CompilerFlags INTERFACE "SIMD_AVX2")
elseif (haveAVX)
	target_compile_options(CompilerFlags INTERFACE
		"$<BUILD_INTERFACE:-mavx>")
	target_compile_definitions(CompilerFlags INTERFACE "SIMD_AVX")
else()
	target_compile_definitions(CompilerFlags INTERFACE "SIMD_NONE")
endif()
