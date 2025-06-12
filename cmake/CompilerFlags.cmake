add_library(CompilerFlags INTERFACE)
target_compile_features(CompilerFlags INTERFACE c_std_17)

# Compiler flags
set(CompilerFlagss "-Wall;-Wextra;-g;-lpthread;-march=native")

# Set compiler flags
target_compile_options(CompilerFlags INTERFACE
	"$<BUILD_INTERFACE:${CompilerFlagss}>")

if (NEED_SPEED)
	message("Using speedy flags")
	target_compile_options(CompilerFlags INTERFACE
		"$<BUILD_INTERFACE:-O3>")
elseif (PROD)
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

if (NO_PYTHON OR NEED_SPEED)
	message("Disabling python interface")
	target_compile_definitions(CompilerFlags INTERFACE "NO_PYTHON")
endif()

if (NO_BOUND_CHECK OR NEED_SPEED)
	message("Disabling bound checking")
	target_compile_definitions(CompilerFlags INTERFACE "NO_BOUND_CHECK")
endif()
if (NO_STATE_CHECK OR NEED_SPEED)
	message("Disabling state checking")
	target_compile_definitions(CompilerFlags INTERFACE "NO_STATE_CHECK")
endif()

if (useAVX512)
	target_compile_options(CompilerFlags INTERFACE
		"$<BUILD_INTERFACE:-mavx512f>")
	target_compile_definitions(CompilerFlags INTERFACE "SIMD_AVX512")
endif()

if (useAVX2)
	target_compile_definitions(CompilerFlags INTERFACE "SIMD_AVX2")
	target_compile_options(CompilerFlags INTERFACE
		"$<BUILD_INTERFACE:-mavx2;-mfma>")
endif()

if (useAVX)
	target_compile_definitions(CompilerFlags INTERFACE "SIMD_AVX")
	target_compile_options(CompilerFlags INTERFACE
		"$<BUILD_INTERFACE:-mavx>")
endif()

if (NOT (useAVX512 AND useAVX2 AND useAVX))
	target_compile_definitions(CompilerFlags INTERFACE "SIMD_NONE")
endif()
