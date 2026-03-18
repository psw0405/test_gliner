cmake_minimum_required(VERSION 3.16)
project(gliner_onnx_inference_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ONNXRUNTIME_DIR)
  message(FATAL_ERROR "Set ONNXRUNTIME_DIR to the extracted ONNX Runtime package path.")
endif()

add_executable(gliner_onnx_inference gliner_onnx_inference.cpp)
target_include_directories(gliner_onnx_inference PRIVATE "${ONNXRUNTIME_DIR}/include")
target_link_directories(gliner_onnx_inference PRIVATE "${ONNXRUNTIME_DIR}/lib")
target_link_libraries(gliner_onnx_inference PRIVATE onnxruntime)

if(WIN32)
  add_custom_command(TARGET gliner_onnx_inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
      $<TARGET_FILE_DIR:gliner_onnx_inference>
  )
endif()
