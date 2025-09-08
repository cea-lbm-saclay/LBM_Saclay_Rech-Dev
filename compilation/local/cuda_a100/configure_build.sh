#!/bin/bash


# find lbm saclay dir

export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export LBM_SACLAY_DIR="$( cd "${SCRIPT_DIR}/../../.."  &> /dev/null && pwd )"


# build dir for this architecture
export BUILD_DIR="$LBM_SACLAY_DIR/build_cuda_a100"

# cmake options
export CMAKE_OPTIONS="-DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON -DUSE_HDF5=ON"



# other operations needed (load modules, export env vars...)
export CC=gcc
export CXX=${LBM_SACLAY_DIR}/external/kokkos/bin/nvcc_wrapper 



# run the cmake configuration
bash $SCRIPT_DIR/../../prepare_cmake.sh





