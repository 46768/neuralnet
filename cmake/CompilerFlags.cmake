add_library(CompilerFlags INTERFACE)
target_compile_features(CompilerFlags INTERFACE c_std_17)

# Compiler flags
set(CompilerFlagss "-Wall;-Wextra;-g;-lpthread")

# Set compiler flags
target_compile_options(CompilerFlags INTERFACE
	"$<BUILD_INTERFACE:${CompilerFlagss}>")

if (PROD)
	message("Using production flags")
	target_compile_options(CompilerFlags INTERFACE
		"$<BUILD_INTERFACE:-O2>")
else()
	message("Using development flags")
	target_compile_options(CompilerFlags INTERFACE
		"$<BUILD_INTERFACE:-Og>")
endif()

if (PROFILING)
	message("Using profiling flags")
	target_compile_options(CompilerFlags INTERFACE
		"$<BUILD_INTERFACE:-pg>")
	target_link_options(CompilerFlags INTERFACE
		"$<BUILD_INTERFACE:-g;-pg>")
endif()

if (NO_PYTHON)
	message("Disabling python interface")
	target_compile_definitions(CompilerFlags INTERFACE "NO_PYTHON")
endif()

if (USE_SCALAR)
	message("Using scalar operations")
	target_compile_definitions(CompilerFlags INTERFACE "SIMD_NONE")
elseif (haveAVX2 AND haveAVX)
	message("Using AVX2 operations")
	target_compile_options(CompilerFlags INTERFACE
		"$<BUILD_INTERFACE:-mavx2;-mfma>")
	target_compile_definitions(CompilerFlags INTERFACE "SIMD_AVX")
	target_compile_definitions(CompilerFlags INTERFACE "SIMD_AVX2")
elseif (haveAVX)
	message("Using AVX operations")
	target_compile_options(CompilerFlags INTERFACE
		"$<BUILD_INTERFACE:-mavx>")
	target_compile_definitions(CompilerFlags INTERFACE "SIMD_AVX")
else()
	message("Using scalar operations")
	target_compile_definitions(CompilerFlags INTERFACE "SIMD_NONE")
endif()
