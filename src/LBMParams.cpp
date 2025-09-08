#include "LBMParams.h"

#include <cstdio>  // for fprintf
#include <cstdlib> // for exit
#include <cstring> // for strcmp
#include <iomanip>
#include <iostream>

#include "utils/config/inih/ini.h" // our INI file reader

#ifdef USE_MPI
using namespace hydroSimu;
#endif // USE_MPI

// =======================================================
// =======================================================
/*
 * LBM Parameters (read parameter file)
 */
void
LBMParams::setup(ConfigMap& configMap)
{
    /* initialize RUN parameters */
    nStepmax = configMap.getInteger("run", "nstepmax", 1000);
    nStepmaxFile = nStepmax;

    max_run_time = configMap.getInteger("run", "max_run_time", 0);
    tEnd = configMap.getFloat("run", "tend", 0.0);
    nOutput = configMap.getInteger("run", "noutput", 100);
    output_selector = OutputSelector(configMap);
    if (nOutput <= 0)
        enableOutput = false;

    set_nStepmax();

    enableProgressInfo = configMap.getBool("run", "enableProgressInfo", false);
    nlog = configMap.getInteger("run", "nlog", 100);

    dt = configMap.getFloat("run", "dt", 1e-4);
    adaptative_dt = configMap.getBool("run", "adaptative_timestep", false);

    std::string lbm_name = configMap.getString("run", "lbm_name", "unknown");

    if (lbm_name == "D2Q5" or lbm_name == "D2Q9")
    {
        dimType = TWO_D;
    }
    else
    {
        dimType = THREE_D;
    }


    // init file params
    initFileName =
      std::string(configMap.getString("init", "data_file", "unknown"));
    sizeRatio = configMap.getFloat("init", "sizeRatio", 1);
    normalizeInputPhi = configMap.getFloat("init", "normalizeInputPhi", 1);
    init_str = std::string(configMap.getString("init", "init_type", "unknown"));

    /* initialize MESH parameters */
    nx = configMap.getInteger("mesh", "nx", 2);
    ny = configMap.getInteger("mesh", "ny", 2);
    nz = configMap.getInteger("mesh", "nz", 1);

    xmin = configMap.getFloat("mesh", "xmin", 0.0);
    ymin = configMap.getFloat("mesh", "ymin", 0.0);
    zmin = configMap.getFloat("mesh", "zmin", 0.0);

    xmax = configMap.getFloat("mesh", "xmax", 1.0);
    ymax = configMap.getFloat("mesh", "ymax", 1.0);
    zmax = configMap.getFloat("mesh", "zmax", 1.0);

    boundary_type_xmin =
      getBoundaryConditionType(configMap, "mesh", "boundary_type_xmin");
    boundary_type_xmax =
      getBoundaryConditionType(configMap, "mesh", "boundary_type_xmax");
    boundary_type_ymin =
      getBoundaryConditionType(configMap, "mesh", "boundary_type_ymin");
    boundary_type_ymax =
      getBoundaryConditionType(configMap, "mesh", "boundary_type_ymax");
    boundary_type_zmin =
      getBoundaryConditionType(configMap, "mesh", "boundary_type_zmin");
    boundary_type_zmax =
      getBoundaryConditionType(configMap, "mesh", "boundary_type_zmax");

    ghostWidth = 2;

    random_seed = configMap.getInteger("init", "random_seed", 54614767);

    init();

    initBoundariesValues(configMap, BOUNDARY_PHASE_FIELD);
    initBoundariesValues(configMap, BOUNDARY_CONCENTRATION);
    initBoundariesValues(configMap, BOUNDARY_CONCENTRATIONB);
    initBoundariesValues(configMap, BOUNDARY_VELOCITY_X);
    initBoundariesValues(configMap, BOUNDARY_VELOCITY_Y);
    initBoundariesValues(configMap, BOUNDARY_VELOCITY_Z);
    initBoundariesValues(configMap, BOUNDARY_PRESSURE);

    initBoundaries(configMap, BOUNDARY_EQUATION_1);
    initBoundaries(configMap, BOUNDARY_EQUATION_2);
    initBoundaries(configMap, BOUNDARY_EQUATION_3);
    initBoundaries(configMap, BOUNDARY_EQUATION_4);

    str2int_t MAP_COLLISION_TYPES = maps::getMAP_COLLISION_TYPES();
    int2str_t MAP_EQS_ID_TO_NAMES = maps::getMAP_EQS_ID_TO_NAMES();

    for (int iEq = 0; iEq < BOUNDARY_COUNT; iEq++)
    {
        std::string eqName = MAP_EQS_ID_TO_NAMES[iEq];
        std::string collisionName =
          configMap.getString(eqName, "collision", "undef");
        if (MAP_COLLISION_TYPES.count(collisionName) >= 1)
        {
            collisionTypes[iEq] = MAP_COLLISION_TYPES[collisionName];
        }
        else
        {
            collisionTypes[iEq] = BGK;
            if (myRank == 0)
                std::cout << "WARNING : Defaulting to BGK for eq " << iEq << std::endl;
        }
    }
    collisionType1 = collisionTypes[BOUNDARY_EQUATION_1];
    collisionType2 = collisionTypes[BOUNDARY_EQUATION_2];
    collisionType3 = collisionTypes[BOUNDARY_EQUATION_3];
    collisionType4 = collisionTypes[BOUNDARY_EQUATION_4];

    lambdaTRT1 = configMap.getFloat("equation1", "lambdaTRT", 1.0);
    lambdaTRT2 = configMap.getFloat("equation2", "lambdaTRT", 1.0);
    lambdaTRT3 = configMap.getFloat("equation3", "lambdaTRT", 1.0);
    lambdaTRT4 = configMap.getFloat("equation4", "lambdaTRT", 1.0);
    TRT_tauS1 = configMap.getFloat("equation1", "tauTRT", 1.0);
    TRT_tauS2 = configMap.getFloat("equation2", "tauTRT", 1.0);
    TRT_tauS3 = configMap.getFloat("equation3", "tauTRT", 1.0);
    TRT_tauS4 = configMap.getFloat("equation4", "tauTRT", 1.0);

    str2int_t MAP_TRT_TAU_METHOD = maps::getMAP_TRT_TAU_METHOD();
    TRT_tauMethod1 = MAP_TRT_TAU_METHOD[configMap.getString(
      "equation1", "tauMethod", "fixed_tau")];
    TRT_tauMethod2 = MAP_TRT_TAU_METHOD[configMap.getString(
      "equation2", "tauMethod", "fixed_tau")];
    TRT_tauMethod3 = MAP_TRT_TAU_METHOD[configMap.getString(
      "equation3", "tauMethod", "fixed_tau")];
    TRT_tauMethod4 = MAP_TRT_TAU_METHOD[configMap.getString(
      "equation4", "tauMethod", "fixed_tau")];

#ifdef USE_MPI
    setup_mpi(configMap);
#endif // USE_MPI

} // LBMParams::setup

#ifdef USE_MPI
// =======================================================
// =======================================================
void
LBMParams::setup_mpi(ConfigMap& configMap)
{

    // runtime determination if we are using float ou double (for MPI
    // communication)
    data_type = typeid(1.0f).name() == typeid((real_t)1.0f).name()
                  ? hydroSimu::MpiComm::FLOAT
                  : hydroSimu::MpiComm::DOUBLE;

    // MPI parameters :
    mx = configMap.getInteger("mpi", "mx", 1);
    my = configMap.getInteger("mpi", "my", 1);
    mz = configMap.getInteger("mpi", "mz", 1);

    // check that parameters are consistent
    bool error = false;
    error |= (mx < 1);
    error |= (my < 1);
    error |= (mz < 1);

    // get world communicator size and check it is consistent with mesh grid sizes
    nProcs = MpiComm::world().getNProc();
    if (nProcs != mx * my * mz)
    {
        std::cerr << "Inconsistent MPI cartesian virtual topology geometry; \n "
                     "mx*my*mz must match with parameter given to mpirun !!!\n";
    }

    // create the MPI communicator for our cartesian mesh
    if (dimType == TWO_D)
    {
        communicator =
          new MpiCommCart(mx, my, MPI_CART_PERIODIC_TRUE, MPI_REORDER_TRUE);
        nDim = 2;
    }
    else
    {
        communicator =
          new MpiCommCart(mx, my, mz, MPI_CART_PERIODIC_TRUE, MPI_REORDER_TRUE);
        nDim = 3;
    }

    // get my MPI rank inside topology
    myRank = communicator->getRank();

    // get my coordinates inside topology
    // myMpiPos[0] is between 0 and mx-1
    // myMpiPos[1] is between 0 and my-1
    // myMpiPos[2] is between 0 and mz-1
    // myMpiPos.resize(nDim);
    {
        int mpiPos[3] = { 0, 0, 0 };
        communicator->getMyCoords(&mpiPos[0]);
        myMpiPos[0] = mpiPos[0];
        myMpiPos[1] = mpiPos[1];
        myMpiPos[2] = mpiPos[2];
    }
    printf("MPI process of Rank %d, is at pos %d, %d, %d \n", myRank, myMpiPos[0], myMpiPos[1], myMpiPos[2]);

    /*
     * compute MPI ranks of our neighbors and
     * set default boundary condition types
     */
    if (dimType == TWO_D)
    {
        nNeighbors = N_NEIGHBORS_2D;
        neighborsRank[X_MIN] = communicator->getNeighborRank<X_MIN>();
        neighborsRank[X_MAX] = communicator->getNeighborRank<X_MAX>();
        neighborsRank[Y_MIN] = communicator->getNeighborRank<Y_MIN>();
        neighborsRank[Y_MAX] = communicator->getNeighborRank<Y_MAX>();
        neighborsRank[Z_MIN] = 0;
        neighborsRank[Z_MAX] = 0;

        neighborsBC[X_MIN] = true;
        neighborsBC[X_MAX] = true;
        neighborsBC[Y_MIN] = true;
        neighborsBC[Y_MAX] = true;
        neighborsBC[Z_MIN] = false;
        neighborsBC[Z_MAX] = false;
        //~ printf("rank %d, X_MIN neighbor %d\n", myRank,neighborsRank[X_MIN]);
        //~ printf("rank %d, X_MAX neighbor %d\n", myRank,neighborsRank[X_MAX]);
        //~ printf("rank %d, Y_MIN neighbor %d\n", myRank,neighborsRank[Y_MIN]);
        //~ printf("rank %d, Y_MAX neighbor %d\n", myRank,neighborsRank[Y_MAX]);

        //~ printf("rank %d, Xpos %d\n", myRank,myMpiPos[0]);
        //~ printf("rank %d, Ypos %d\n", myRank,myMpiPos[1]);
        //~ printf("rank %d, Zpos %d\n", myRank,myMpiPos[2]);
        //~ printf("rank %d, xmin %g, xmax %g\n", myRank,xmin,xmax);
    }
    else
    {
        nNeighbors = N_NEIGHBORS_3D;
        neighborsRank[X_MIN] = communicator->getNeighborRank<X_MIN>();
        neighborsRank[X_MAX] = communicator->getNeighborRank<X_MAX>();
        neighborsRank[Y_MIN] = communicator->getNeighborRank<Y_MIN>();
        neighborsRank[Y_MAX] = communicator->getNeighborRank<Y_MAX>();
        neighborsRank[Z_MIN] = communicator->getNeighborRank<Z_MIN>();
        neighborsRank[Z_MAX] = communicator->getNeighborRank<Z_MAX>();

        neighborsBC[X_MIN] = true;
        neighborsBC[X_MAX] = true;
        neighborsBC[Y_MIN] = true;
        neighborsBC[Y_MAX] = true;
        neighborsBC[Z_MIN] = true;
        neighborsBC[Z_MAX] = true;
    }

    /*
     * identify outside boundaries (no actual communication if we are
     * doing BC_ZERO_FLUX_AT_NODE for example)
     *
     * Please notice the duality
     * XMIN -- boundary_xmax
     * XMAX -- boundary_xmin
     *
     */

    // X_MIN boundary
    if (myMpiPos[DIR_X] == 0)
    {
        neighborsBC[X_MIN] = (boundary_type_xmin == BC_PERIODIC);
        //~ printf("rank %d, boundary X_MIN %d\n", myRank, neighborsBC[X_MIN]);
    }
    // X_MAX boundary
    if (myMpiPos[DIR_X] == mx - 1)
    {
        neighborsBC[X_MAX] = (boundary_type_xmax == BC_PERIODIC);
        //~ printf("rank %d, boundary X_MAX %d\n", myRank, neighborsBC[X_MAX]);
    }

    // Y_MIN boundary
    if (myMpiPos[DIR_Y] == 0)
    {
        neighborsBC[Y_MIN] = (boundary_type_ymax == BC_PERIODIC);
        //~ printf("rank %d, boundary Y_MIN %d\n", myRank, neighborsBC[Y_MIN]);
    }
    // Y_MAX boundary
    if (myMpiPos[DIR_Y] == my - 1)
    {
        neighborsBC[Y_MAX] = (boundary_type_ymin == BC_PERIODIC);
        //~ printf("rank %d, boundary Y_MAX %d\n", myRank, neighborsBC[Y_MAX]);
    }
    if (dimType == THREE_D)
    {

        // Z_MIN boundary
        if (myMpiPos[DIR_Z] == 0)
        {
            neighborsBC[Z_MIN] = (boundary_type_zmin == BC_PERIODIC);
        }
        // Y_MAX boundary
        if (myMpiPos[DIR_Z] == mz - 1)
        {
            neighborsBC[Z_MAX] = (boundary_type_zmax == BC_PERIODIC);
        }
    } // end THREE_D

    mpiBounds(BOUNDARY_EQUATION_1);
    mpiBounds(BOUNDARY_EQUATION_2);
    mpiBounds(BOUNDARY_EQUATION_3);
    mpiBounds(BOUNDARY_EQUATION_4);

    /*
     * Initialize CUDA device if needed.
     *
     * Let's assume hwloc is doing its job !
     *
     * Old comments from RamsesGPU:
     * When running on a Linux machine with mutiple GPU per node, it might be
     * very helpfull if admin has set the CUDA device compute mode to exclusive
     * so that a device is only attached to 1 host thread (i.e. 2 different host
     * thread can not communicate with the same GPU).
     *
     * As a sys-admin, just run for all devices command:
     *   nvidia-smi -g $(DEV_ID) -c 1
     *
     * If compute mode is set to normal mode, we need to use cudaSetDevice,
     * so that each MPI device is mapped onto a different GPU device.
     *
     * At CCRT, on machine Titane, each node (2 quadri-proc) "sees" only
     * half a Tesla S1070, that means cudaGetDeviceCount should return 2.
     * If we want the ration 1 MPI process <-> 1 GPU, we need to allocate
     * N nodes and 2*N tasks (MPI process).
     */
#ifdef KOKKOS_ENABLE_CUDA
    // // get device count
    // int count;
    // cutilSafeCall( cudaGetDeviceCount(&count) );

    // int devId = myRank % count;
    // cutilSafeCall( cudaSetDevice(devId) );

    // cudaDeviceProp deviceProp;
    // int myDevId = -1;
    // cutilSafeCall( cudaGetDevice( &myDevId ) );
    // cutilSafeCall( cudaGetDeviceProperties( &deviceProp, myDevId ) );
    // // faire un cudaSetDevice et cudaGetDeviceProp et aficher le nom
    // // ajouter un booleen dans le constructeur pour savoir si on veut faire ca
    // // sachant que sur Titane, probablement que le mode exclusif est active
    // // a verifier demain

    // std::cout << "MPI process " << myRank << " is using GPU device num " <<
    // myDevId << std::endl;

#endif // KOKKOS_ENABLE_CUDA

    // fix space resolution :
    // need to take into account number of MPI process in each direction
    dx = (xmax - xmin) / (nx * mx);
    dy = (ymax - ymin) / (ny * my);
    dz = (zmax - zmin) / (nz * mz);

    // print information about current setup
    if (myRank == 0)
    {
        std::cout << "We are about to start simulation with the following "
                     "characteristics\n";

        std::cout << "Global resolution : " << nx * mx << " x " << ny * my << " x " << nz * mz << "\n";
        std::cout << "Local  resolution : " << nx << " x " << ny << " x " << nz << "\n";
        std::cout << "MPI Cartesian topology : " << mx << "x" << my << "x" << mz << std::endl;
    }

} // LBMParams::setup_mpi

// =======================================================
// =======================================================
void
LBMParams::mpiBounds(BoundaryField FieldId)
{
    const int gw = ghostWidth;
    bounds[FieldId][FACE_XMIN] = (neighborsBC[X_MIN]) ? 0 : gw;
    bounds[FieldId][FACE_XMAX] = (neighborsBC[X_MAX]) ? isize - 1 : isize - gw - 1;
    bounds[FieldId][FACE_YMIN] = (neighborsBC[Y_MIN]) ? 0 : gw;
    bounds[FieldId][FACE_YMAX] = (neighborsBC[Y_MAX]) ? jsize - 1 : jsize - gw - 1;
    bounds[FieldId][FACE_ZMIN] = (neighborsBC[Z_MIN]) ? 0 : gw;
    bounds[FieldId][FACE_ZMAX] = (neighborsBC[Z_MAX]) ? ksize - 1 : ksize - gw - 1;
}

#endif // USE_MPI

// =======================================================
// =======================================================
void
LBMParams::initBoundariesValues(ConfigMap& configMap,
                                BoundaryFieldValues FieldId)
{
    std::string field_to_make_boundary;

    if (FieldId == BOUNDARY_PHASE_FIELD)
        field_to_make_boundary = "phase_field_boundary";
    else if (FieldId == BOUNDARY_CONCENTRATION)
        field_to_make_boundary = "concentration_boundary";
    else if (FieldId == BOUNDARY_CONCENTRATIONB)
        field_to_make_boundary = "cB_boundary";
    else if (FieldId == BOUNDARY_VELOCITY_X)
        field_to_make_boundary = "vx_boundary";
    else if (FieldId == BOUNDARY_VELOCITY_Y)
        field_to_make_boundary = "vy_boundary";
    else if (FieldId == BOUNDARY_VELOCITY_Z)
        field_to_make_boundary = "vz_boundary";
    else if (FieldId == BOUNDARY_PRESSURE)
        field_to_make_boundary = "pressure_boundary";

    boundary_values[FieldId][FACE_XMIN] = configMap.getFloat(field_to_make_boundary, "boundary_value_xmin", 0.0);
    boundary_values[FieldId][FACE_XMAX] = configMap.getFloat(field_to_make_boundary, "boundary_value_xmax", 0.0);
    boundary_values[FieldId][FACE_YMIN] = configMap.getFloat(field_to_make_boundary, "boundary_value_ymin", 0.0);
    boundary_values[FieldId][FACE_YMAX] = configMap.getFloat(field_to_make_boundary, "boundary_value_ymax", 0.0);
    boundary_values[FieldId][FACE_ZMIN] = configMap.getFloat(field_to_make_boundary, "boundary_value_zmin", 0.0);
    boundary_values[FieldId][FACE_ZMAX] = configMap.getFloat(field_to_make_boundary, "boundary_value_zmax", 0.0);

} // LBMParams::initBoundaryTypes

// =======================================================
// =======================================================
void
LBMParams::initBoundaries(ConfigMap& configMap, BoundaryField FieldId)
{
    std::string field_to_make_boundary;

    if (FieldId == BOUNDARY_EQUATION_1)
        field_to_make_boundary = "equation1";
    else if (FieldId == BOUNDARY_EQUATION_2)
        field_to_make_boundary = "equation2";
    else if (FieldId == BOUNDARY_EQUATION_3)
        field_to_make_boundary = "equation3";
    else if (FieldId == BOUNDARY_EQUATION_4)
        field_to_make_boundary = "equation4";

    boundary_types[FieldId][FACE_XMIN] = getBoundaryConditionType(configMap, field_to_make_boundary, "boundary_type_xmin");
    boundary_types[FieldId][FACE_XMAX] = getBoundaryConditionType(configMap, field_to_make_boundary, "boundary_type_xmax");
    boundary_types[FieldId][FACE_YMIN] = getBoundaryConditionType(configMap, field_to_make_boundary, "boundary_type_ymin");
    boundary_types[FieldId][FACE_YMAX] = getBoundaryConditionType(configMap, field_to_make_boundary, "boundary_type_ymax");
    boundary_types[FieldId][FACE_ZMIN] = getBoundaryConditionType(configMap, field_to_make_boundary, "boundary_type_zmin");
    boundary_types[FieldId][FACE_ZMAX] = getBoundaryConditionType(configMap, field_to_make_boundary, "boundary_type_zmax");

    const int gw = ghostWidth;

    bounds[FieldId][FACE_XMIN] =
      (boundary_types[FieldId][FACE_XMIN] == BC_PERIODIC) ? 0 : gw;
    bounds[FieldId][FACE_XMAX] =
      (boundary_types[FieldId][FACE_XMAX] == BC_PERIODIC) ? isize - 1
                                                          : isize - gw - 1;
    bounds[FieldId][FACE_YMIN] =
      (boundary_types[FieldId][FACE_YMIN] == BC_PERIODIC) ? 0 : gw;
    bounds[FieldId][FACE_YMAX] =
      (boundary_types[FieldId][FACE_YMAX] == BC_PERIODIC) ? jsize - 1
                                                          : jsize - gw - 1;
    bounds[FieldId][FACE_ZMIN] =
      (boundary_types[FieldId][FACE_ZMIN] == BC_PERIODIC) ? 0 : gw;
    bounds[FieldId][FACE_ZMAX] =
      (boundary_types[FieldId][FACE_ZMAX] == BC_PERIODIC) ? ksize - 1
                                                          : ksize - gw - 1;

    // std::cout << "Boundary types for " << field_to_make_boundary << ":" << std::endl;
    // std::cout << "XMIN : " << boundary_types[FieldId][FACE_XMIN] << std::endl;
    // std::cout << "XMAX : " << boundary_types[FieldId][FACE_XMAX] << std::endl;
    // std::cout << "YMIN : " << boundary_types[FieldId][FACE_YMIN] << std::endl;
    // std::cout << "YMAX : " << boundary_types[FieldId][FACE_YMAX] << std::endl;
    // std::cout << "ZMIN : " << boundary_types[FieldId][FACE_ZMIN] << std::endl;
    // std::cout << "ZMAX : " << boundary_types[FieldId][FACE_ZMAX] << std::endl;
}
// =======================================================
// =======================================================
Kokkos::Array<int, 6>
LBMParams::getBounds(BoundaryField FieldId)
{
    return bounds[FieldId];
} // LBMParams::getBounds

// =======================================================
// =======================================================
BoundaryConditionType
LBMParams::getBoundaryConditionType(ConfigMap& configMap,
                                    std::string field_to_make_boundary,
                                    std::string boundary)
{
    BoundaryConditionType type_to_return;
    std::string boundary_type_name =
      configMap.getString(field_to_make_boundary, boundary, "undef");
    str2int_t BC_NAMES_TO_ID = maps::getMAP_BC_NAMES_TO_ID();
    if (BC_NAMES_TO_ID.count(boundary_type_name) == 1)
    {
        type_to_return =
          static_cast<BoundaryConditionType>(BC_NAMES_TO_ID[boundary_type_name]);
    }
    else
    {
        type_to_return = BC_UNDEFINED;
    }

    return type_to_return;
}

void
LBMParams::set_nStepmax()
{
    while (nStepmax % nOutput != 1 and nOutput > 1)
    {
        nStepmax++;
    }
    if (nStepmax != nStepmaxFile && myRank == 0)
    {
        printf("WARNING: nStepmax changed to %d \n", nStepmax); // to be =1[nOut] and not waste calculations.
    }
};
// =======================================================
// =======================================================
void
LBMParams::init()
{

    // set other parameters
    imax = nx - 1 + 2 * ghostWidth;
    jmax = ny - 1 + 2 * ghostWidth;
    kmax = nz - 1 + 2 * ghostWidth;

    isize = imax - imin + 1;
    jsize = jmax - jmin + 1;
    ksize = kmax - kmin + 1;

    dx = (xmax - xmin) / nx;
    dy = (ymax - ymin) / ny;
    dz = (zmax - zmin) / nz;

} // LBMParams::init

// =======================================================
// =======================================================
void
LBMParams::print()
{

    printf("##########################\n");
    printf("Simulation run parameters:\n");
    printf("##########################\n");
    printf("nx         : %d | ny         : %d | nz         : %d\n", nx, ny, nz);
    printf("ghostWidth : %d\n", ghostWidth);
    printf("dx         : %f | dy         : %f | dz         : %f\n", dx, dy, dz);
    printf("imin       : %d | imax       : %d\n", imin, imax);
    printf("jmin       : %d | jmax       : %d\n", jmin, jmax);
    printf("kmin       : %d | kmax       : %d\n", kmin, kmax);
    printf("dt         : %f\n", dt);
    printf("nStepmax   : %d\n", nStepmax);
    printf("tEnd       : %f\n", tEnd);
    printf("nOutput    : %d\n", nOutput);
    printf("##########################\n");

} // LBMParams::print
