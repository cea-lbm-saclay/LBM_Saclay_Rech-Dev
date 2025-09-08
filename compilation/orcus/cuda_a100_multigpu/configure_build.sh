#!/bin/bash


# find lbm saclay dir

export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export LBM_SACLAY_DIR="$( cd "${SCRIPT_DIR}/../../.."  &> /dev/null && pwd )"


# build dir for this architecture
export BUILD_DIR="$LBM_SACLAY_DIR/build_cuda_a100_multigpu"

# cmake options
export CMAKE_OPTIONS="-DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_AMPERE80=ON -DUSE_MPI=ON -DUSE_MPI_CUDA_AWARE_ENFORCED=ON -DKokkos_ENABLE_HWLOC=ON -DUSE_HDF5=ON -DCMAKE_SKIP_RPATH=ON -DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE"



# other operations needed (load modules, export env vars...)
source $SCRIPT_DIR/../modules



# run the cmake configuration
bash $SCRIPT_DIR/../../prepare_cmake.sh





