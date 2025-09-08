/**
 * \file LBMParams.h
 * \brief LBM solver parameters.
 *
 * \date April, 27 2018
 */
#ifndef LBM_PARAMS_H_
#define LBM_PARAMS_H_

#include "kokkos_shared.h"
#include "real_type.h"

#include "LBM_enums.h"
#include "LBM_maps.h"

#include "ConfigMap.h"
#include "OutputSelector.h"

#include <stdbool.h>
#include <unordered_map>
#ifdef USE_MPI
#include "utils/mpiUtils/MpiCommCart.h"
#endif // USE_MPI

//! a convenience alias to map variable names to id
using str2int_t = std::unordered_map<std::string, int>;

/*
 * LBM Parameters (declaration)
 */
struct LBMParams {
#ifdef USE_MPI
    using MpiCommCart = hydroSimu::MpiCommCart;
#endif // USE_MPI

    // run parameters
    int nStepmax; /*!< maximun number of time steps. */
    int nStepmaxFile; /*!< maximun number of time steps indicated in ini file, to compare. */
    int nStepmaxPrev; /*!< number of iterations already done in case of restart. Initialized automatically.*/
    
    /** 
     * \brief Maximum real time LBM_saclay will run before interrupting the simulation (in seconds).
     * 
     * Use 0 to set no limit (default).
     * Useful on clusters where execution time is limited, to ensure Final output 
     * will be written with the necessary data for a restart.
     */
    int max_run_time;
    
    real_t tEnd; /*!< end of simulation time. */
    int nOutput; /*!< number of time steps between 2 consecutive outputs. */
    
    OutputSelector output_selector; /*!< contains rules to select steps to output */
    int nlog; /*!<  number of time steps between 2 consecutive logs. */
    bool enableOutput; /*!< enable output file write. */
    bool enableProgressInfo; /*!< enable output file write. */
    real_t dt;
    bool adaptative_dt;

    DimensionType dimType; //!< 2D or 3D.

    // init file params
    std::string initFileName;
    int sizeRatio;
    real_t normalizeInputPhi;
    std::string init_str;
    // geometry parameters
    int nx; /*!< logical size along X (without ghost cells).*/
    int ny; /*!< logical size along Y (without ghost cells).*/
    int nz; /*!< logical size along Z (without ghost cells).*/
    int ghostWidth;
    int imin; /*!< index minimum at X border*/
    int imax; /*!< index maximum at X border*/
    int jmin; /*!< index minimum at Y border*/
    int jmax; /*!< index maximum at Y border*/
    int kmin; /*!< index minimum at Z border*/
    int kmax; /*!< index maximum at Z border*/

    int isize; /*!< total size (in cell unit) along X direction with ghosts.*/
    int jsize; /*!< total size (in cell unit) along Y direction with ghosts.*/
    int ksize; /*!< total size (in cell unit) along Z direction with ghosts.*/

    real_t xmin;
    real_t xmax;
    real_t ymin;
    real_t ymax;
    real_t zmin;
    real_t zmax;
    real_t dx; /*!< x resolution */
    real_t dy; /*!< y resolution */
    real_t dz; /*!< z resolution */

    BoundaryConditionType boundary_type_xmin;
    BoundaryConditionType boundary_type_xmax;
    BoundaryConditionType boundary_type_ymin;
    BoundaryConditionType boundary_type_ymax;
    BoundaryConditionType boundary_type_zmin;
    BoundaryConditionType boundary_type_zmax;

    Kokkos::Array<Kokkos::Array<BoundaryConditionType, 6>, BOUNDARY_COUNT> boundary_types;
    Kokkos::Array<Kokkos::Array<int, 6>, BOUNDARY_COUNT> bounds;
    Kokkos::Array<Kokkos::Array<real_t, 6>, BOUNDARY_VALUES_COUNT> boundary_values;

    // IO parameters
    bool ioVTK; /*!< enable VTK  output file format (using VTI).*/
    bool ioHDF5; /*!< enable HDF5 output file format.*/

    int random_seed;

    Kokkos::Array<int, BOUNDARY_COUNT> collisionTypes;
    int collisionType1, collisionType2, collisionType3, collisionType4;
    real_t lambdaTRT1, lambdaTRT2, lambdaTRT3, lambdaTRT4;
    real_t TRT_tauS1, TRT_tauS2, TRT_tauS3, TRT_tauS4;
    real_t TRT_tauMethod1, TRT_tauMethod2, TRT_tauMethod3, TRT_tauMethod4;

    //! MPI rank of current process
    int myRank; // to be valid and useable both with MPI or without

#ifdef USE_MPI
    //! runtime determination if we are using float ou double (for MPI communication)
    //! initialized in constructor to either MpiComm::FLOAT or MpiComm::DOUBLE
    int data_type;

    //! size of the MPI cartesian grid
    int mx, my, mz;

    //! MPI communicator in a cartesian virtual topology
    MpiCommCart* communicator;

    //! number of dimension
    int nDim;

    //! number of MPI processes
    int nProcs;

    //! MPI cartesian coordinates inside MPI topology
    Kokkos::Array<int, 3> myMpiPos;

    //! number of MPI process neighbors (4 in 2D and 6 in 3D)
    int nNeighbors;

    //! MPI rank of adjacent MPI processes
    Kokkos::Array<int, 6> neighborsRank;

    //! boundary condition type with adjacent domains (corresponding to
    //! neighbor MPI processes)
    Kokkos::Array<bool, 6> neighborsBC;

#endif // USE_MPI
    LBMParams()
        : nStepmax(0)
        , nStepmaxFile(0)
        , nStepmaxPrev(0)
        , tEnd(0.0)
        , nOutput(0)
        , nlog(100)
        , enableOutput(true)
        , enableProgressInfo(false)
        , nx(0)
        , ny(0)
        , nz(0)
        , ghostWidth(1)
        , imin(0)
        , imax(0)
        , jmin(0)
        , jmax(0)
        , kmin(0)
        , kmax(0)
        , isize(0)
        , jsize(0)
        , ksize(0)
        , xmin(0.0)
        , xmax(1.0)
        , ymin(0.0)
        , ymax(1.0)
        , zmin(0.0)
        , zmax(1.0)
        , dx(0.0)
        , dy(0.0)
        , dz(0.0)
        , boundary_type_xmin(BC_UNDEFINED)
        , boundary_type_xmax(BC_UNDEFINED)
        , boundary_type_ymin(BC_UNDEFINED)
        , boundary_type_ymax(BC_UNDEFINED)
        , boundary_type_zmin(BC_UNDEFINED)
        , boundary_type_zmax(BC_UNDEFINED)
        , ioVTK(true)
        , ioHDF5(false)
        , myRank(0) {

        };

    //! This is the genuine initialization / setup (fed by parameter file)
    void setup(ConfigMap& map);

    void initBoundariesValues(ConfigMap& configMap, BoundaryFieldValues FieldId);
    void initBoundaries(ConfigMap& configMap, BoundaryField EquationId);

    BoundaryConditionType getBoundaryConditionType(ConfigMap& configMap, std::string field_to_make_boundary, std::string boundary);

    Kokkos::Array<int, 6> getBounds(BoundaryField FieldId);

    void set_nStepmax();

#ifdef USE_MPI
    //! Initialize MPI-specific parameters
    void setup_mpi(ConfigMap& map);
    void mpiBounds(BoundaryField FieldId);
#endif // USE_MPI

    void init();
    void print();

}; // struct LBMParams

#endif // LBM_PARAMS_H_
