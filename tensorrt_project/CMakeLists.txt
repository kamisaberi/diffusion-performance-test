
cmake_minimum_required(VERSION 3.18)
project(TensorRTDiffusion LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
add_executable(tensorrt_infer inference_tensorrt.cpp)
target_include_directories(tensorrt_infer PRIVATE ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(tensorrt_infer PRIVATE ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES})
set_property(TARGET tensorrt_infer PROPERTY CXX_STANDARD 17)