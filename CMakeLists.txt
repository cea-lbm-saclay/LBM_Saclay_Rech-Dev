# set minimal version the one requested by kokkos
cmake_minimum_required(VERSION 3.10)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0")
  message(STATUS "Setting policy CMP0076 to use new behavior")
  cmake_policy(SET CMP0076 NEW)
endif()

project(LBM_saclay LANGUAGES C CXX)

#
# Make sure external dependecies are available
#
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/kokkos/Makefile.kokkos" OR
        NOT EXISTS "${PROJECT_SOURCE_DIR}/external/backward-cpp/CMakeLists.txt")
  execute_process(
    COMMAND git submodule update --init --recursive
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()


include(GNUInstallDirs)

#
# default local cmake macro repository
#
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

#
# Prevent from build in source tree
#
include(preventBuildInSource)

#
# Init build type: Release (default), Debug, ...
#
include(initBuildType)

#
# Main options
#
option (USE_MPI "Activate / want MPI build" OFF)
option (USE_DOUBLE "build with double precision" ON)
option (USE_HDF5 "build HDF5 input/output support" OFF)
option (USE_PNETCDF "build PNETCDF input/output support (MPI required)" OFF)
option (USE_FPE_DEBUG "build with floating point Nan tracing (signal handler)" OFF)
option (USE_MPI_CUDA_AWARE_ENFORCED "Some MPI cuda-aware implementation are not well detected; use this to enforce" OFF)


#
# problem selection options
#
set(PROBLEM "NSAC_coupling" CACHE STRING "directory for problem definition")
set(PROBLEMDIR "${PROBLEM}")


# generate content for problem_include header file
set(PBM_INCLUDES_CONTENT "#ifndef PBM_INCLUDES_H_\n")
string(APPEND PBM_INCLUDES_CONTENT "#define PBM_INCLUDES_H_\n")
foreach(entry IN LISTS PROBLEMDIR)
  string(APPEND PBM_INCLUDES_CONTENT "#include \"${PROJECT_SOURCE_DIR}/src/kernels/${entry}/Setup.h\"\n")
endforeach()
string(APPEND PBM_INCLUDES_CONTENT "#endif //PBM_INCLUDES_H_")

file(GENERATE 
    OUTPUT ${PROJECT_BINARY_DIR}/src/kernels/Problem_includes.h
    CONTENT ${PBM_INCLUDES_CONTENT}
)
include_directories(${PROJECT_BINARY_DIR}/src/kernels)


#
# documentation related options
#
option(LBM_SACLAY_DOC_ENABLED  "Enable / disable documentation build" OFF)

# documentation type - the only valid values are : doxygen and mkdocs
set(LBM_SACLAY_DOC_TYPE "doxygen" CACHE STRING "documentation type (doxygen or mkdocs)")

# always export compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#
# Write a small header with build data, git version, etc...
#
include(write_version)

# C++11 is for Kokkos
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

#####################################################################
# External packages: MPI, ...
#####################################################################

#####################################################################
#
#~ find_package(MPI)
if (USE_MPI)
  find_package(MPI REQUIRED)
  if(MPI_CXX_FOUND)
    message(STATUS "MPI support found")
    message(STATUS "MPI compile flags: " ${MPI_CXX_COMPILE_FLAGS})
    message(STATUS "MPI include path: " ${MPI_CXX_INCLUDE_PATH})
    message(STATUS "MPI LINK flags path: " ${MPI_CXX_LINK_FLAGS})
    message(STATUS "MPI libraries: " ${MPI_CXX_LIBRARIES})

    #set(CMAKE_EXE_LINKER_FLAGS ${MPI_CXX_LINK_FLAGS})
    find_program(OMPI_INFO
      NAMES ompi_info
      HINTS ${MPI_CXX_LIBRARIES}/../bin)

    # Full command line to probe if cuda support in MPI implementation is enabled
    # ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
    if (OMPI_INFO)
      execute_process(COMMAND ${OMPI_INFO}
        OUTPUT_VARIABLE _output)
      if ( (_output MATCHES "smcuda") OR (USE_MPI_CUDA_AWARE_ENFORCED) )
        message(STATUS "Found OpenMPI with CUDA support built.")
      else()
        message(WARNING "OpenMPI found, but it is not built with CUDA support.")
    add_compile_options(-DMPI_CUDA_AWARE_OFF)
      endif()
    endif()
  else()
    message(WARNING "Not compiling with MPI. Suppress this warning with -DUSE_MPI=OFF")
    set(USE_MPI OFF)
  endif()
endif()

#####################################################################
# HDF5
#####################################################################
# prefer using parallel HDF5 when build with mpi
if (USE_MPI)
  set(HDF5_PREFER_PARALLEL TRUE)
endif(USE_MPI)

if (USE_HDF5)
  find_package(HDF5)
  if(HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIRS})
    set(my_hdf5_libs hdf5 hdf5_cpp)
    add_compile_options(-DUSE_HDF5 -DH5_NO_DEPRECATED_SYMBOLS)
    if (HDF5_IS_PARALLEL)
      add_compile_options(-DUSE_HDF5_PARALLEL)
    endif()
  endif(HDF5_FOUND)
endif(USE_HDF5)

#####################################################################
# PNETCDF
#####################################################################
if (USE_MPI)
  if (USE_PNETCDF)
    find_package(PNETCDF)
    if (PNETCDF_FOUND)
      add_compile_options(-DUSE_PNETCDF)
      include_directories(${PNETCDF_INCLUDE_DIRS})
    endif(PNETCDF_FOUND)
  endif(USE_PNETCDF)
endif(USE_MPI)

#
# common flags
#

if (USE_DOUBLE)
  add_compile_options(-DUSE_DOUBLE)
endif()

if (USE_MPI)
  add_compile_options(-DUSE_MPI)
endif()

if (USE_FPE_DEBUG)
  add_compile_options(-DUSE_FPE_DEBUG)
endif()

# for prettier stacktrace log with backward-cpp
add_definitions( -g )
# let the compiler warn us about anything
add_definitions( -Wall )

# (May 7 2019) deactivated backward until it is compatible
# with cuda 10.1 again
#add_subdirectory(external/backward-cpp)
add_subdirectory(external/kokkos)

# pass Kokkos include directories to our target application
include_directories(${Kokkos_INCLUDE_DIRS_RET})

#add_subdirectory(test)
add_subdirectory(src)


add_custom_target(partial-clean
    COMMAND ${CMAKE_BUILD_TOOL} clean
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src
    COMMENT "Cleaning only non kokkos part of the build"
)

if(LBM_SACLAY_DOC_ENABLED)
  add_subdirectory(doc)
endif()

##################### PRINT CONFIGURE STATUS ######################
message("//===================================================")
message("  ${PROJECT_NAME} build configuration:")
message("//===================================================")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
                         "${CMAKE_CXX_COMPILER_VERSION} "
                         "${CMAKE_CXX_COMPILER_WRAPPER}")
message("    ${CMAKE_CXX_COMPILER}")
if (USE_MPI)
  message("  MPI enabled")
else(USE_MPI)
  message("  MPI not enabled")
endif(USE_MPI)
message("  Kokkos OpenMP enabled : ${Kokkos_ENABLE_OPENMP}")
message("  Kokkos CUDA   enabled : ${Kokkos_ENABLE_CUDA}")
if (Kokkos_ENABLE_CUDA)
  message("  Kokkos CUDA   Lambda  : ${Kokkos_ENABLE_CUDA_LAMBDA}")
  message("  Kokkos CUDA   flags   : ${KOKKOS_CUDA_OPTIONS}")
endif(Kokkos_ENABLE_CUDA)
message("  Kokkos HWLOC  enabled : ${Kokkos_ENABLE_HWLOC}")
if (HDF5_FOUND)
  message("  HDF5 found version    : ${HDF5_VERSION}")
  message("  HDF5 definitions      : ${HDF5_DEFINITIONS}")
  message("  HDF5 parallel         : ${HDF5_IS_PARALLEL}")
  message("  HDF5 includes dirs    : ${HDF5_INCLUDE_DIRS}")
  message("  HDF5 libraries        : ${HDF5_LIBRARIES}")
endif(HDF5_FOUND)
if (LBM_SACLAY_DOC_ENABLED)
  message("   Building doc of type ${LBM_SACLAY_DOC_TYPE}")
endif (LBM_SACLAY_DOC_ENABLED)
message("")
