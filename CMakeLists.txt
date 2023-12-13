cmake_minimum_required(VERSION 3.22)
project(vector_add C)
set(CMAKE_C_STANDARD 99)

add_compile_options(-Wall -O2 -DCL_TARGET_OPENCL_VERSION=300)
add_executable(vector_add vector_add.c)
target_link_libraries(vector_add -lOpenCL)