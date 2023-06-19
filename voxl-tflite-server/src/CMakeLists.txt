cmake_minimum_required(VERSION 3.3)

SET(TARGET voxl-tflite-server)

option(BUILD_QRB5165 "Build the qrb5165 binary" OFF)

if(BUILD_QRB5165)
    add_definitions(-DBUILD_QRB5165)
endif()

# Enable compile optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Enable debug flags (use if you want to debug in gdb)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -Wmaybe-uninitialized -fno-omit-frame-pointer")

# Build from all source files
file(GLOB all_src_files *.c*)

add_executable(${TARGET}
	${all_src_files}
)

include_directories(
    ../include
    /usr/include/opencv4/

    # # apq8096 SPECIFIC
    # /usr/include/tensorflow/lite/tools/make/downloads/absl/
    # /usr/include/tensorflow/lite/tools/make/downloads/flatbuffers/include/

    # qrb5165 SPECIFIC
    /usr/include/flatbuffers/include/
    /usr/include/abseil-cpp/
    /usr/include/ruy/
)

set(ABSL_LIBS
    "absl_bad_any_cast_impl"
    "absl_bad_optional_access"
    "absl_bad_variant_access"
    "absl_base"
    "absl_city"
    "absl_civil_time"
    "absl_cord"
    "absl_debugging_internal"
    "absl_demangle_internal"
    "absl_exponential_biased"
    "absl_flags"
    "absl_flags_commandlineflag"
    "absl_flags_commandlineflag_internal"
    "absl_flags_config"
    "absl_flags_internal"
    "absl_flags_marshalling"
    "absl_flags_private_handle_accessor"
    "absl_flags_program_name"
    "absl_flags_reflection"
    "absl_graphcycles_internal"
    "absl_hash"
    "absl_hashtablez_sampler"
    "absl_int128"
    "absl_log_severity"
    "absl_malloc_internal"
    "absl_raw_hash_set"
    "absl_raw_logging_internal"
    "absl_spinlock_wait"
    "absl_stacktrace"
    "absl_status"
    "absl_str_format_internal"
    "absl_strings"
    "absl_strings_internal"
    "absl_symbolize"
    "absl_synchronization"
    "absl_throw_delegate"
    "absl_time"
    "absl_time_zone"
    "absl_wyhash"
)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-implicit-fallthrough")
set(LINK_LIBS
    "tensorflow-lite"
    ${ABSL_LIBS}
    ${ABSL_LIBS}
    ${ABSL_LIBS}
    "flatbuffers"
    "cpuinfo"
    "farmhash"
    "clog"
    "fft2d_fftsg2d"
    "fft2d_fftsg"
    "ruy_allocator"
    "ruy_apply_multiplier"
    "ruy_block_map"
    "ruy_blocking_counter"
    "ruy_context"
    "ruy_context_get_ctx"
    "ruy_cpuinfo"
    "ruy_ctx"
    "ruy_denormal"
    "ruy_frontend"
    "ruy_have_built_path_for_avx"
    "ruy_have_built_path_for_avx2_fma"
    "ruy_have_built_path_for_avx512"
    "ruy_kernel_arm"
    "ruy_kernel_avx"
    "ruy_kernel_avx2_fma"
    "ruy_kernel_avx512"
    "ruy_pack_arm"
    "ruy_pack_avx"
    "ruy_pack_avx2_fma"
    "ruy_pack_avx512"
    "ruy_prepacked_cache"
    "ruy_prepare_packed_matrices"
    "ruy_profiler_instrumentation"
    "ruy_system_aligned_alloc"
    "ruy_thread_pool"
    "ruy_trmul"
    "ruy_tune"
    "ruy_wait"
    "glib-2.0"
    "dl"
    "stdc++"
    "pthread"
    "z"
    "cutils"
    "log"
    "sync"
    "gthread-2.0"
    "pcre"
    "modal_pipe"
    "modal_json"
    "opencv_core"
    "opencv_highgui"
    "opencv_imgproc"
    "opencv_imgcodecs"
    "gsl"
    "llvm-qcom"
    "adreno_utils"
    "OpenCL"
    "CB"
    "EGL_adreno"
    "GLESv2_adreno"
    "rt"
)

if (BUILD_QRB5165)
set(LINK_LIBS "${LINK_LIBS}"
    "XNNPACK"
    "pthreadpool"
)
endif()

target_link_libraries(${TARGET}
    -Wl,-rpath-link,/usr/lib64/
    -L/usr/lib64/
    ${LINK_LIBS}
)
# make sure everything is installed where we want
# LIB_INSTALL_DIR comes from the parent cmake file
install(
	TARGETS			${TARGET}
	LIBRARY			DESTINATION ${LIB_INSTALL_DIR}
	RUNTIME			DESTINATION /usr/bin
	PUBLIC_HEADER	DESTINATION /usr/include
)
