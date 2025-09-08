/**
 * \file LBMRun.h
 */
#ifndef LBM_RUN_H_
#define LBM_RUN_H_

#include <map> // for std::map
#include <memory> // for std::unique_ptr / std::shared_ptr
#include <unordered_map> // for std::map
#include <filesystem> // for std::filesystem

#include "kokkos_shared.h"

#include "LBMParams.h"

#include "LBMRunBase.h" // base class needed for building a factory
#include "LBM_Base_Functor.h"
#include "LBM_Lattice.h"

//#include "finite_diffs.h"

#include "io/IO_ReadWrite.h"
#include "io/io_common.h"

#ifdef USE_HDF5
#include "io/IO_HDF5.h" // HDF5 output
#endif // USE_HDF5

#include <fstream>
#include <sstream> // std::stringstream

// limit conditions
#include "bc_cond/PeriodicBoundariesManager.h"

// the actual computational functors called in LBMRun
#include "kernels/Kernels.h"
#include "kernels/ProblemFactory.h"

// for MPI border
#ifdef USE_MPI
#include "bc_cond/mpiBorderUtils.h"
#endif // USE_MPI

// for timer
#ifdef KOKKOS_ENABLE_CUDA
#include "utils/monitoring/CudaTimer.h"
#else
#include "utils/monitoring/OpenMPTimer.h"
#endif

enum TimerIds {
    TIMER_TOTAL,
    TIMER_IO,
    TIMER_BOUNDARIES,
    TIMER_MPICOMMS,
    TIMER_UPDATE_M,
    TIMER_UPDATE_F,
    TIMER_UPDATE_TIMESTEP,
    TIMER_INIT_MACRO,
    TIMER_INIT_FDIST,
    TIMER_OUTPUT,
    COUNTER // invalid index to count timers
}; // enum TimerIds

/**
 * Main LBM structure.
 *
 * \note LBMArray data is not really needed since density and velocity are
 * obtained as moments of the distribution function, and as such can be
 * recomputed as needed. Later we could provide implementation where those
 * variables are not stored in memory.
 *
 * \note We use two distribution functions here, because in the first
 * implementation, there are used to stored current distribution function,
 * and equilibrium dist. function. In the PUSH ou PULL implementation these
 * arrays are used to stored current and next time steps dist. function.
 *
 * \tparam dim should be 2 or 3
 * \tparam npop is the number of population of our lattice
 */
template <int dim, int npop>
class LBMRun : public LBMRunBase {
    //! alias to data array storing of physics variables (density, velocity, ...)
    using LBMArray = typename LBMBaseFunctor<dim, npop>::LBMArray;
    using LBMArrayHost = typename LBMArray::HostMirror;

    //! this is just an alias but makes a difference between physics variables and populations variables
    using FArray = typename std::conditional<dim == 2, FArray2d, FArray3d>::type;
    
    
    //! retrieve number of space dimension (2 or 3)
    static constexpr int get_dim()
    {
        return dim;
    };
    //! retrieve number of population (aka number of distribution functions)
    static constexpr int get_npop()
    {
        return npop;
    };

public:
    real_t get_time()
    {
        return m_t;
    };

    /**
     * Static creation method called by the LBMrun factory.
     */
    static LBMRunBase* create(LBMParams& params, ConfigMap& configMap)
    {
        LBMRun<dim, npop>* lbm = new LBMRun<dim, npop>(params, configMap);

        return lbm;
    } // create

    LBMRun(LBMParams& params, ConfigMap& configMap)
        : params(params)
        , configMap(configMap)
    {

        // create the timers
        timers[TIMER_TOTAL] = std::make_shared<Timer>();
        timers[TIMER_IO] = std::make_shared<Timer>();
        timers[TIMER_BOUNDARIES] = std::make_shared<Timer>();
        timers[TIMER_MPICOMMS] = std::make_shared<Timer>();
        timers[TIMER_UPDATE_M] = std::make_shared<Timer>();
        timers[TIMER_UPDATE_F] = std::make_shared<Timer>();
        timers[TIMER_INIT_MACRO] = std::make_shared<Timer>();
        timers[TIMER_INIT_FDIST] = std::make_shared<Timer>();
        timers[TIMER_OUTPUT] = std::make_shared<Timer>();

        //initialize geometry.
        m_t = 0.0; /* TODO : change that when doing a re-start */
        m_dt = params.dt;

        m_times_saved = 0;
        m_step = 0;
        myRank = 0;
        lastOutputTimeElapsed = 0;
        lastOutputTimeInterval = 0;
#ifdef USE_MPI
        myRank = params.myRank;
#endif

        // setup the rigth pbm depending on ini file
        setup_pbm();

    } // LBMRun::LBMRun

    virtual ~LBMRun()
    {
        delete ptr2pbm;
    };

    // =======================================================
    // ==== CLASS LBMRun setup pbm ===========================
    // =======================================================

    void setup_pbm()
    {
        
        ProblemFactory<dim, npop>::Instance().enumerate_options();

        std::string problemStr = std::string(configMap.getString("lbm", "problem", "unknown"));
        std::string modelStr = std::string(configMap.getString("lbm", "model", "unknown"));

        if (myRank == 0) {
            std::cout << "Problem chosen: " << problemStr << std::endl;
            std::cout << "Model chosen  : " << modelStr << std::endl;
            printf("##########################\n");
        }
        //
        try {

            ptr2pbm=ProblemFactory<dim, npop>::Instance().create(problemStr, modelStr, params, configMap);

        } catch (const std::exception& e) {

            std::cout << e.what() << std::endl;
            exit(1);
        }
    }

    // =======================================================
    // ==== CLASS LBMRun simulation methods ==================
    // =======================================================
    /** Initialize the simulation, then start the main loop*/
    void run()
    {

        timers[TIMER_TOTAL]->start();
        timers[TIMER_OUTPUT]->start();
        if (myRank == 0) {
            std::cout << "starting init" << std::endl;
        }
        init_wrapper();
        if (myRank == 0) {
            std::cout << "Start computation.... LBM D" << dim << "Q" << npop << std::endl;
            std::cout << "use the following command to force stopping the simulation and still get the final output for restart" << std::endl;
            std::cout << "touch " << std::filesystem::current_path().append("LBMstop") << std::endl;
        }
        // LBM solver loop
        while (m_t < params.tEnd && m_step < params.nStepmax) {
            // output infologs and simulation data
            output();
            // perform one time step
            update_wrapper();

            // increase time
            m_step++;
            m_t += m_dt;
            ptr2pbm->update_total_time(m_t);

            // update timestep
            if (params.adaptative_dt) {
                timers[TIMER_UPDATE_TIMESTEP]->start();
                ptr2pbm->update_dt();
                m_dt = ptr2pbm->get_dt();
                timers[TIMER_UPDATE_TIMESTEP]->stop();
            }
            
            
            // check whether stopping 
            
            // if (std::filesystem::exists("LBMstop")) {
            //     printf("Finishing simulation due to LBMstop file\n");
            //     std::cout << std::flush;
            //     std::filesystem::remove("LBMstop");
            //     break;
            // }
            
            // check simulation time limit
            timers[TIMER_TOTAL]->stop();
            if (params.max_run_time > 0 and timers[TIMER_TOTAL]->elapsed() > params.max_run_time) {
                printf("Finishing simulation due to maximum simulation time reached\n");
                std::cout << std::flush;
                break;
            }
            timers[TIMER_TOTAL]->start();

        } // end solver loop

        // output all fields of the last iteration to allow for restart
        if (myRank == 0) {
            printf("Output results at final time\n");
            std::cout << std::flush;
        }
        
        ptr2pbm->hook_before_output(m_step);
        ptr2pbm->hook_before_final_output();
        timers[TIMER_IO]->start();
        timers[TIMER_OUTPUT]->start();
        ptr2pbm->writeDataFull(m_step, m_t, m_times_saved);
        timers[TIMER_IO]->stop();
        timers[TIMER_OUTPUT]->stop();

        timers[TIMER_TOTAL]->stop();

        // print monitoring information
        if (myRank == 0) {
            print_monitoring_information();
        }

        // write Xdmf wrapper file if necessary
#ifdef USE_HDF5
        bool outputHdf5Enabled = configMap.getBool("output", "hdf5_enabled", false);
        if (outputHdf5Enabled) {
            lbm_saclay::writeXdmfForHdf5Wrapper(params, configMap, ptr2pbm->get_var_to_write(), m_times_saved - 1, false);
        }
#endif // USE_HDF5
    } // LBMRun<dim,npop>::run

    // =======================================================
    // =======================================================
    /** wrapper for using the pbm class to initialize*/
    void init_wrapper()
    {
        timers[TIMER_INIT_MACRO]->start();
        std::string initType = configMap.getString("init", "init_type", "");
        if (initType == "restart") {
            ptr2pbm->loadDataFull(m_step, m_t, m_times_saved);
            params.nStepmaxPrev = m_step;
            params.nStepmax = params.nStepmaxFile + m_step;
            params.set_nStepmax();
            params.tEnd += m_t;
            //~ m_times_saved=m_step/params.nOutput+1;
        } else {
            ptr2pbm->init_m();
        }
        timers[TIMER_INIT_MACRO]->stop();
        timers[TIMER_INIT_FDIST]->start();
        ptr2pbm->init_f();
        timers[TIMER_INIT_FDIST]->stop();
        //~ timers[TIMER_BOUNDARIES]->start();
        //~ ptr2pbm->make_boundaries();
        //~ timers[TIMER_BOUNDARIES]->stop();
    }

    // =======================================================
    // =======================================================
    /** wrapper for using the pbm class to update*/
    void update_wrapper()
    {
        timers[TIMER_UPDATE_F]->start();
        ptr2pbm->update_f();
        timers[TIMER_UPDATE_F]->stop();
        timers[TIMER_BOUNDARIES]->start();
        ptr2pbm->make_boundaries();
        timers[TIMER_BOUNDARIES]->stop();
        timers[TIMER_MPICOMMS]->start();
        ptr2pbm->make_periodic_boundaries(); // not always done in this function, can be done in the prvious one
        timers[TIMER_MPICOMMS]->stop();
        timers[TIMER_UPDATE_M]->start();
        ptr2pbm->update_m();
        timers[TIMER_UPDATE_M]->stop();
    }

    // =======================================================
    // =======================================================
    /** output*/
    void output()
    {

        timers[TIMER_OUTPUT]->stop();

        // log
        if ((myRank == 0) and (m_step> 0) and (m_step % params.nlog == 0) and not(params.output_selector.is_output_step(m_step))) {
            printf("Log    results at step %6d | time elapsed %9.2fs\n", m_step, timers[TIMER_OUTPUT]->elapsed());
        }

        // output
        if (params.enableOutput and ( params.output_selector.is_output_step(m_step))) {

            ptr2pbm->hook_before_output(m_step);
            //~ ptr2pbm->bcPerMgr.print_timers();
            //~ ptr2pbm->bcPerMgr.reset_timers();



            real_t timeb = timers[TIMER_IO]->elapsed();
            timers[TIMER_IO]->start();
            timers[TIMER_OUTPUT]->start();
            m_times_saved++;
            ptr2pbm->writeData(m_step, m_t, m_times_saved);
            
#ifdef USE_HDF5
            bool outputHdf5Enabled = configMap.getBool("output", "hdf5_enabled", false);
            if (outputHdf5Enabled) {
                lbm_saclay::writeXdmfForHdf5Wrapper(params, configMap, ptr2pbm->get_var_to_write(), m_times_saved - 1, false);
            }
#endif // USE_HDF5

            timers[TIMER_IO]->stop();
            timers[TIMER_OUTPUT]->stop();
            real_t timea = timers[TIMER_IO]->elapsed();


            if (myRank == 0) {
                lastOutputTimeInterval = (timers[TIMER_OUTPUT]->elapsed() - lastOutputTimeElapsed);
                lastOutputTimeElapsed = (timers[TIMER_OUTPUT]->elapsed());
                printf("Output results at step %6d | time elapsed %9.2fs | written in %6.2fs\n", m_step, lastOutputTimeElapsed, (timea - timeb));
                std::cout << std::flush;
            }

        } // end enable output
        if (params.enableOutput and params.enableProgressInfo and not(params.output_selector.is_output_step(m_step))) {
            if (myRank == 0) {
                real_t prog = ((1.0 * (m_step % params.nOutput)) / params.nOutput);
                real_t eta1 = ((1.0 - prog) * lastOutputTimeInterval);
                real_t currentTimeToLastOutput = (timers[TIMER_OUTPUT]->elapsed() - lastOutputTimeElapsed);
                real_t eta2 = ((1.0 - prog) * currentTimeToLastOutput / prog);
                real_t eta = FMAX(eta1, eta2);
                printf("Next output : %5.2f%%          | eta : %6.2fs", (100.0 * prog), eta);
                std::cout << '\r' << std::flush;
            }
        }
        timers[TIMER_OUTPUT]->start();

    } // LBMRun<dim,npop>::output

    // =======================================================
    // =======================================================
    /**  print monitoring information */
    void print_monitoring_information()
    {

        real_t t_tot = timers[TIMER_TOTAL]->elapsed();
        real_t t_update_f = timers[TIMER_UPDATE_F]->elapsed();
        real_t t_update_m = timers[TIMER_UPDATE_M]->elapsed();
        real_t t_bound = timers[TIMER_BOUNDARIES]->elapsed();
        real_t t_mpi = timers[TIMER_MPICOMMS]->elapsed();
        real_t t_io = timers[TIMER_IO]->elapsed();
        real_t t_init_macro = timers[TIMER_INIT_MACRO]->elapsed();
        real_t t_init_fdist = timers[TIMER_INIT_FDIST]->elapsed();

        real_t t_sum = t_update_f + t_update_m + t_bound + t_io + t_init_macro + t_init_fdist;
        real_t t_util = t_tot - t_io;
        real_t t_compute = t_tot - t_io - t_init_macro;

        int isize = params.isize;
        int jsize = params.jsize;
        int ksize = params.ksize;

        int64_t nbcells = dim == 2 ? isize * jsize : isize * jsize * ksize;

        int nProcs = 1;
#ifdef USE_MPI
        nProcs = params.nProcs;
#endif // USE_MPI

        int steps = m_step - params.nStepmaxPrev;
        
        int nbEqs = ptr2pbm->get_nbEqs();
        int nbVar = ptr2pbm->get_nbVar();

        printf("total          time : %6.3f seconds\n", t_tot);
        printf("init macro     time : %6.3f seconds\n", t_init_macro);
        printf("init fdist     time : %6.3f seconds\n", t_init_fdist);
        printf("lbm kernels    time : %6.3f seconds %5.2f%%\n", t_update_f, 100 * t_update_f / t_tot);
        printf("macro update   time : %6.3f seconds %5.2f%%\n", t_update_m, 100 * t_update_m / t_tot);
        printf("boundaries     time : %6.3f seconds %5.2f%%\n", t_bound, 100 * t_bound / t_tot);
        printf("mpi comms      time : %6.3f seconds %5.2f%%\n", t_mpi, 100 * t_mpi / t_tot);
        printf("io             time : %6.3f seconds %5.2f%%\n", t_io, 100 * t_io / t_tot);
        printf("sum (~=tot)    time : %6.3f seconds\n", t_sum);
        printf("util (tot-io)  time : %6.3f seconds\n", t_util);
        printf("device         time : %6.3f seconds\n", t_compute);
        // print perf as update per second per million mesh node (scales down with number of eqs/var, because more computation per cell)
        printf("Perf                : %10.2f number of Mcell-updates/s\n", 1.0 * steps * nbcells * nProcs * nbEqs / (t_compute)*1e-6);

        /*
         * print memory bandwidth usage (per MPI process)
         * to evaluate memory bandwidth usage at node level.
         */
        // compute total amount of memory read and write (effective) in bytes
        // per MPI process

        
        // read access :
        // collision : for each cell, reads (npop + nbVar) * nbEqs
        // update macro : for each cell, reads  (npop * nbEqs)
        int readPerCell = (2 * npop + nbVar) * nbEqs + nbVar;
        real_t totalBytesRead = 1.0 * nProcs * nbcells * readPerCell * sizeof(real_t) * steps;

        // write access :
        // stream : for each cell, writes (npop) * nbEqs
        // update macro : for each cell, writes  (nbVar)
        int writePerCell = npop * nbEqs + nbVar;
        real_t totalBytesWrite = 1.0 * nProcs * nbcells * writePerCell * sizeof(real_t) * steps;

        // print total bandwidth
        real_t totalBytes = totalBytesRead + totalBytesWrite;
        printf("Memory performance per MPI process (based on device usage time):\n");
        printf("Read/Write per lattice node : %d/%d\n", readPerCell, writePerCell);
        printf("Memory bandwidth    : %10.2f Gbytes/s\n", 1e-9 * totalBytes / (t_compute));
        
        
        real_t total_memory_used = 1e-9 * ((nbEqs+1)*npop + nbVar)* nbcells * sizeof(real_t);
        
        printf("Memory allocated : %10.2f Gbytes\n", total_memory_used);

    } // LBMRun<dim,npop>::print_monitoring_info;

    // =======================================================
    // ==== CLASS LBMRun attributes ==========================
    // =======================================================

    LBMParams& params;
    ConfigMap& configMap;

    ProblemBase<dim, npop>* ptr2pbm;

    real_t m_t; /*!< total time */
    real_t m_dt; /*!< time step */

    int m_step; /*!< number of steps */
    int myRank; /*!< rank of mpi process */
    int m_times_saved; /*!< number of output */

    real_t lastOutputTimeElapsed;
    real_t lastOutputTimeInterval;

#ifdef KOKKOS_ENABLE_CUDA
    using Timer = CudaTimer;
#else
    using Timer = OpenMPTimer;
#endif
    using TimerMap = std::map<int, std::shared_ptr<Timer>>;
    TimerMap timers;

    // =======================================================
    // ==== End LBMRun attributes ==========================
    // =======================================================

}; // class LBMRun

#endif // LBM_RUN_H_
