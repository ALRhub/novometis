# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# CUDA17 dialect is only supported in CMake 3.18 and above
# https://forums.developer.nvidia.com/t/build-cuda-11-with-c-17-msvc-nvcc-cmake/154858/3
# https://github.com/colmap/colmap/issues/2581
cmake_minimum_required(VERSION 3.22)
project(torch_isolation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_subdirectory("pinocchio_isolation/")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# Find Torch
execute_process(
    COMMAND python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    OUTPUT_VARIABLE site_packages_path
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(Torch_DIR "${site_packages_path}/torch/share/cmake/Torch/")
find_package(Torch REQUIRED)

# Find Eigen
if(NOT TARGET Eigen3)
    find_package(Eigen3 REQUIRED)
endif()

# Define common include directories
set(COMMON_INCLUDE_DIRS 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

add_library(torch_server_ops SHARED src/torch_server_ops.cpp)
set_target_properties(torch_server_ops PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_include_directories(torch_server_ops PUBLIC ${COMMON_INCLUDE_DIRS})
target_link_libraries(torch_server_ops PRIVATE
    ${TORCH_LIBRARIES}
)

add_library(torchrot SHARED src/rotations.cpp)
set_target_properties(torchrot PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_include_directories(torchrot PUBLIC ${COMMON_INCLUDE_DIRS})
target_link_libraries(torchrot PRIVATE
    ${TORCH_LIBRARIES}
)

add_library(torchscript_pinocchio SHARED src/pinocchio.cpp)
set_target_properties(torchscript_pinocchio PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_include_directories(torchscript_pinocchio PUBLIC ${COMMON_INCLUDE_DIRS})
target_link_libraries(torchscript_pinocchio PRIVATE
    pinocchio_wrapper
    ${TORCH_LIBRARIES}
    torchrot
)

# by default, set the install prefix to the conda environment
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}" CACHE PATH "Default install path" FORCE)
endif()

# installs to $CONDA_PREFIX/lib/libtorchscript_pinocchio.so by default
install(TARGETS torchrot torchscript_pinocchio DESTINATION lib)
