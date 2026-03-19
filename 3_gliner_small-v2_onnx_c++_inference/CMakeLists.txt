cmake_minimum_required(VERSION 3.16)
project(gliner_onnx_inference_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ONNXRUNTIME_DIR)
  if(DEFINED ENV{ONNXRUNTIME_DIR})
    set(ONNXRUNTIME_DIR "$ENV{ONNXRUNTIME_DIR}")
  else()
    message(FATAL_ERROR "Set ONNXRUNTIME_DIR (or environment variable ONNXRUNTIME_DIR) to the extracted ONNX Runtime package path.")
  endif()
endif()

if(NOT EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "ONNX Runtime header not found: ${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
endif()

if(NOT EXISTS "${ONNXRUNTIME_DIR}/lib")
  message(FATAL_ERROR "ONNX Runtime lib directory not found: ${ONNXRUNTIME_DIR}/lib")
endif()

if(WIN32)
  set(ONNXRUNTIME_LINK_LIB "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
  set(ONNXRUNTIME_RUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll")
elseif(UNIX AND NOT APPLE)
  set(ONNXRUNTIME_LINK_LIB "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
else()
  message(FATAL_ERROR "Unsupported platform for this project. Use Windows or Linux.")
endif()

if(NOT EXISTS "${ONNXRUNTIME_LINK_LIB}")
  message(FATAL_ERROR "ONNX Runtime link library not found: ${ONNXRUNTIME_LINK_LIB}")
endif()

add_executable(gliner_onnx_inference gliner_onnx_inference.cpp)
target_include_directories(gliner_onnx_inference PRIVATE "${ONNXRUNTIME_DIR}/include")
target_link_libraries(gliner_onnx_inference PRIVATE "${ONNXRUNTIME_LINK_LIB}")

if(UNIX AND NOT APPLE)
  # Keep runtime search path so the executable can load libonnxruntime.so
  # from ONNXRUNTIME_DIR/lib without requiring global linker configuration.
  set_target_properties(gliner_onnx_inference PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
  )
endif()

if(WIN32)
  if(NOT EXISTS "${ONNXRUNTIME_RUNTIME_LIB}")
    message(FATAL_ERROR "ONNX Runtime DLL not found: ${ONNXRUNTIME_RUNTIME_LIB}")
  endif()

  add_custom_command(TARGET gliner_onnx_inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ONNXRUNTIME_RUNTIME_LIB}"
      $<TARGET_FILE_DIR:gliner_onnx_inference>
  )
endif()
