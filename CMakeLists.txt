cmake_minimum_required(VERSION 4.3)
project(referencebook)

set(CMAKE_CXX_STANDARD 20)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an ONNX Runtime installation")

find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime/onnxruntime_cxx_api.h
        HINTS "${ONNXRUNTIME_ROOT}" "$ENV{ONNXRUNTIME_ROOT}"
        PATH_SUFFIXES include)
find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        HINTS "${ONNXRUNTIME_ROOT}" "$ENV{ONNXRUNTIME_ROOT}"
        PATH_SUFFIXES lib lib64)

if (NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
            "ONNX Runtime not found. Set ONNXRUNTIME_ROOT or provide "
            "ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIBRARY.")
endif ()

add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED)
set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")

add_executable(referencebook src/main.cpp)
target_link_libraries(referencebook PRIVATE ONNXRuntime::ONNXRuntime)
