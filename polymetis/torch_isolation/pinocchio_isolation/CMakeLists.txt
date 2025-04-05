# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
cmake_minimum_required(VERSION 3.14)
project(pinocchio_isolation)

SET(CMAKE_CXX_FLAGS "-std=c++14 -O3")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

# Find spdlog (C++ logger)
if(NOT TARGET spdlog)
    find_package(spdlog REQUIRED)
endif()

# Eigen requires Boost, but conda's Boost does not install BoostConfig.cmake
set(Boost_NO_BOOST_CMAKE ON)  # forces CMake to not search for BoostConfig.cmake and fall back to its FindBoost.cmake module
set(BOOST_ROOT $ENV{CONDA_PREFIX})

# Find Eigen
find_package(Eigen3 REQUIRED)

# Find Pinocchio
set(pinocchio_DIR $ENV{CONDA_PREFIX}/lib/cmake/pinocchio)
find_package(pinocchio REQUIRED)

# Path to the pinocchio library is not set correctly in the pinocchioConfig.cmake file
set_target_properties(pinocchio::pinocchio PROPERTIES
    IMPORTED_LOCATION "$ENV{CONDA_PREFIX}/lib/libpinocchio.so"
)

message(STATUS "Pinocchio include dirs: ${pinocchio_INCLUDE_DIRS}")
message(STATUS "Pinocchio library dirs: ${pinocchio_LIBRARIES}")

get_target_property(pinocchio_location pinocchio::pinocchio IMPORTED_LOCATION)
message(STATUS "Pinocchio library location: ${pinocchio_location}")

get_target_property(pinocchio_link_libraries pinocchio::pinocchio INTERFACE_LINK_LIBRARIES)
message(STATUS "Pinocchio linked libraries: ${pinocchio_link_libraries}")

# Define pinocchio_wrapper library
add_library(pinocchio_wrapper SHARED src/pinocchio_wrapper.cpp)
set_target_properties(pinocchio_wrapper PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_include_directories(pinocchio_wrapper
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${pinocchio_INCLUDE_DIRS}
)
target_link_libraries(pinocchio_wrapper
    pinocchio::pinocchio
    spdlog::spdlog
)
