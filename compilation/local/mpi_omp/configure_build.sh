#!/bin/bash


# find lbm saclay dir

export SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export LBM_SACLAY_DIR="$( cd "${SCRIPT_DIR}/../../.."  &> /dev/null && pwd )"


# build dir for this architecture
export BUILD_DIR="$LBM_SACLAY_DIR/build_mpi_omp"

# cmake options
export CMAKE_OPTIONS="-DKokkos_ENABLE_OPENMP=ON -DUSE_MPI=ON -DUSE_HDF5=ON"



# other operations needed (load modules, export env vars...)
# nothing in this case



# run the cmake configuration
bash $SCRIPT_DIR/../../prepare_cmake.sh





