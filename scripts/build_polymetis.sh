#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

GIT_ROOT=$(git rev-parse --show-toplevel)
POLYMETIS_PATH="$GIT_ROOT/polymetis"

# Check to make sure directory exists
[ ! -d $POLYMETIS_PATH ] && echo "Directory $POLYMETIS_PATH does not exist" && exit 1

# Build
BUILD_PATH="${POLYMETIS_PATH}/build"
if [ -d "$BUILD_PATH" ]; then rm -r $BUILD_PATH; fi
mkdir -p $BUILD_PATH && cd $BUILD_PATH
echo "Building polymetis at $BUILD_PATH"

# set up the correct path for building

export CONDA_PREFIX=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CPATH=${CONDA_PREFIX}/include
export LIBRARY_PATH=${CONDA_PREFIX}/lib
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib

# Need to add -DNDEBUG Flag to CXX because of grpc/abseil https://github.com/abseil/abseil-cpp/issues/1624#issuecomment-1968073823
# cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-DNDEBUG -DBUILD_FRANKA=OFF -DBUILD_SERVER=ON -DBUILD_TESTS=OFF ..
# cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-DNDEBUG -DBUILD_FRANKA=ON -DBUILD_SERVER=OFF -DBUILD_TESTS=OFF ..
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-DNDEBUG -DBUILD_FRANKA=ON -DBUILD_SERVER=ON -DBUILD_TESTS=OFF ..
# cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-DNDEBUG -fsanitize=address -fno-omit-frame-pointer" -DBUILD_FRANKA=ON -DBUILD_SERVER=ON -DBUILD_TESTS=OFF ..
cmake --build . -j --target install

cd -


pip install -e ./polymetis