cmake_minimum_required(VERSION 3.18)
project(orpheus-speak LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Dependencies ──────────────────────────────────────────────────────────────
# libcurl  – HTTP client for llama-server API
find_package(CURL REQUIRED)

# ONNX Runtime – SNAC decoder inference
# Set ONNXRUNTIME_ROOT to the extracted onnxruntime release directory, e.g.:
#   cmake -DONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-gpu-1.18.0 ..
if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR
        "Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime  (download from "
        "https://github.com/microsoft/onnxruntime/releases)")
endif()

find_library(ONNXRUNTIME_LIB onnxruntime
    PATHS ${ONNXRUNTIME_ROOT}/lib
    REQUIRED)

# ── Build ─────────────────────────────────────────────────────────────────────
add_executable(orpheus-speak
    orpheus-speak.cpp
)

target_include_directories(orpheus-speak PRIVATE
    ${ONNXRUNTIME_ROOT}/include
)

target_link_libraries(orpheus-speak PRIVATE
    CURL::libcurl
    ${ONNXRUNTIME_LIB}
)

# Install
install(TARGETS orpheus-speak RUNTIME DESTINATION bin)
install(PROGRAMS speak.sh DESTINATION bin)
