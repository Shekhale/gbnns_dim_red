cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)
find_package(Torch REQUIRED)

add_executable(search_with_torch /home/shekhale/dim-red/search_with_torch.cpp)
target_link_libraries(search_with_torch "${TORCH_LIBRARIES}")
SET( CMAKE_CXX_FLAGS "-Ofast -DNDEBUG -std=c++11 -DHAVE_CXX0X -fopenmp -march=native -fpic -w -ftree-vectorize" )
set_property(TARGET search_with_torch PROPERTY CXX_STANDARD 14)
