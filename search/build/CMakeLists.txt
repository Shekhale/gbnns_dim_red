cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)
find_package(Torch REQUIRED)

add_executable(cpu_only_test /home/shekhale/gbnns_dim-red/search/cpu_only_test.cpp)
target_link_libraries(cpu_only_test "${TORCH_LIBRARIES}")
SET( CMAKE_CXX_FLAGS "-Ofast -DNDEBUG -std=c++11 -DHAVE_CXX0X -fopenmp -march=native -fpic -w -ftree-vectorize" )
set_property(TARGET cpu_only_test PROPERTY CXX_STANDARD 14)
