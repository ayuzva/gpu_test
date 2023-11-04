project(GpuTest
    LANGUAGES CUDA CXX
)

cmake_minimum_required(VERSION 3.0)

add_executable(gpu_test main.cu)