/**
 * LBM with kokkos miniapp.
 *
 * \date April, 28 2018
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "kokkos_shared.h"

#include "LBMParams.h" // read parameter file
#include "LBMRun.h" // memory allocation for hydro arrays
#include "LBMRunFactory.h"
#include "real_type.h" // choose between single and double precision

// banner
#include "LBM_saclay_version.h"

#ifdef USE_MPI
#include "utils/mpiUtils/GlobalMpiSession.h"
#include <mpi.h>
#endif // USE_MPI

// ===============================================================
// ===============================================================
void run_lbm(int argc, char* argv[])
{
    [[maybe_unused]] int rank = 0;
    [[maybe_unused]] int nRanks = 1;

#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
#endif

    // read parameter file and initialize parameter
    // parse parameters from input file
    std::string input_file = std::string(argv[1]);
    ConfigMap configMap(input_file);

    // test: create a LBMParams object
    LBMParams params = LBMParams();
    params.setup(configMap);

    // print parameters on screen
    if (rank == 0)
        params.print();

    // retrieve LBM type name from settings
    // default value is unknow to force the setting
    // to explicitely state this parameter
    const std::string lbm_name = configMap.getString("run", "lbm_name", "Unknown");

    // initialize workspace memory
    //auto *lbm = new LBMRun<2,NPOP_D2Q5>(params, configMap);
    //auto *lbm = new LBMRun<2,NPOP_D2Q9>(params, configMap);
    //auto *lbm = new LBMRun<3,NPOP_D3Q7>(params, configMap);
    auto* lbm = LBMRunFactory::Instance().create(lbm_name, params, configMap);

    // main computation done here
    lbm->run();

    if (rank == 0)
        printf("final time is %f\n", lbm->get_time());

    delete lbm;

} // run_lbm

// ===============================================================
// ===============================================================
// ===============================================================
int main(int argc, char* argv[])
{

    // Create MPI session if MPI enabled
#ifdef USE_MPI
    hydroSimu::GlobalMpiSession mpiSession(&argc, &argv);
#endif // USE_MPI

    /*
     * Initialize kokkos (host + device)
     *
     * If CUDA is enabled, Kokkos will try to use the default GPU,
     * i.e. GPU #0 if you have multiple GPUs.
     */
    Kokkos::initialize(argc, argv);

    [[maybe_unused]] int rank = 0;
    [[maybe_unused]] int nRanks = 1;


#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
#endif
    if (rank == 0) {

        std::cout << "##########################\n";
        std::cout << "KOKKOS CONFIG             \n";
        std::cout << "##########################\n";

        std::ostringstream msg;
        std::cout << "Kokkos configuration" << std::endl;
        if (Kokkos::hwloc::available()) {
            msg << "hwloc( NUMA[" << Kokkos::hwloc::get_available_numa_count()
                << "] x CORE[" << Kokkos::hwloc::get_available_cores_per_numa()
                << "] x HT[" << Kokkos::hwloc::get_available_threads_per_core()
                << "] )"
                << std::endl;
        }
        Kokkos::print_configuration(msg);
        std::cout << msg.str();
        std::cout << "##########################\n";

    } // end

#ifdef KOKKOS_ENABLE_CUDA
    // verbose log MPI/GPU mapping
    {

        // on a large cluster, the scheduler should assign ressources
        // in a way that each MPI task is mapped to a different GPU
        // let's cross-checked that:

        int cudaDeviceId;
        cudaGetDevice(&cudaDeviceId);
        std::cout << "I'm MPI task #" << rank << " (out of " << nRanks << ")"
                  << " pinned to GPU #" << cudaDeviceId << "\n";
    }
#endif // KOKKOS_ENABLE_CUDA

    // banner
    if (rank == 0)
        print_version_info();

    // chekc command line arguments
    if (argc != 2) {
        if (rank == 0)
            fprintf(stderr, "Error: wrong number of argument; input filename must be the only parameter on the command line\n");
        exit(EXIT_FAILURE);
    }

    run_lbm(argc, argv);

    Kokkos::finalize();

    return EXIT_SUCCESS;

} // end main
