#ifndef PERIODIC_MANAGER_H_
#define PERIODIC_MANAGER_H_

#include "LBMParams.h"
#include "LBM_Base_Functor.h"
#include "PeriodicBoundariesFunctor.h"
#include "mpiBorderUtils.h"

#ifdef KOKKOS_ENABLE_CUDA
#include "utils/monitoring/CudaTimer.h"
#else
#include "utils/monitoring/OpenMPTimer.h"
#endif

enum PeriodicTimerIds {
    TIMER_TOTAL_PERIODIC,
    TIMER_COPY,
    TIMER_TRANSFERT,
    TIMER_COPY_BACK,
}; // enum TimerIds

template <int dim, int npop>
struct PeriodicManager {

    using FArray = typename LBMBaseFunctor<dim, npop>::FArray;
    using FState = typename Kokkos::Array<real_t, npop>;

    LBMParams params;

#ifdef KOKKOS_ENABLE_CUDA
    using Timer = CudaTimer;
#else
    using Timer = OpenMPTimer;
#endif
    using TimerMap = std::map<int, std::shared_ptr<Timer>>;
    TimerMap timers;

    // constructor for 2 equation (ie 2 lbm distribution of type FArray)
    PeriodicManager(LBMParams params)
        : params(params)
    {

        // create the timers
        timers[TIMER_TOTAL_PERIODIC] = std::make_shared<Timer>();
        timers[TIMER_COPY] = std::make_shared<Timer>();
        timers[TIMER_TRANSFERT] = std::make_shared<Timer>();
        timers[TIMER_COPY_BACK] = std::make_shared<Timer>();

#ifdef USE_MPI
        const int isize = params.isize;
        const int jsize = params.jsize;
        const int ksize = params.ksize;
        const int gw = params.ghostWidth;
        // Please note that for MOOD or SDM or any other scheme that uses a different
        // number of per cell, the following border buffer will have to be RESIZED

        if (dim == 2) {
            borderBufSend_xmin_2d = FArray2d("borderBufSend_xmin_2d", gw, jsize, npop);
            borderBufSend_xmax_2d = FArray2d("borderBufSend_xmax_2d", gw, jsize, npop);
            borderBufSend_ymin_2d = FArray2d("borderBufSend_ymin_2d", isize, gw, npop);
            borderBufSend_ymax_2d = FArray2d("borderBufSend_ymax_2d", isize, gw, npop);

            borderBufRecv_xmin_2d = FArray2d("borderBufRecv_xmin", gw, jsize, npop);
            borderBufRecv_xmax_2d = FArray2d("borderBufRecv_xmax", gw, jsize, npop);
            borderBufRecv_ymin_2d = FArray2d("borderBufRecv_ymin", isize, gw, npop);
            borderBufRecv_ymax_2d = FArray2d("borderBufRecv_ymax", isize, gw, npop);


            borderBufSend_xmin_2d_host = Kokkos::create_mirror_view(borderBufSend_xmin_2d);
            borderBufSend_xmax_2d_host = Kokkos::create_mirror_view(borderBufRecv_xmax_2d);
            borderBufRecv_xmin_2d_host = Kokkos::create_mirror_view(borderBufSend_xmin_2d);
            borderBufRecv_xmax_2d_host = Kokkos::create_mirror_view(borderBufRecv_xmax_2d);

            borderBufSend_ymin_2d_host = Kokkos::create_mirror_view(borderBufSend_ymin_2d);
            borderBufSend_ymax_2d_host = Kokkos::create_mirror_view(borderBufRecv_ymax_2d);
            borderBufRecv_ymin_2d_host = Kokkos::create_mirror_view(borderBufSend_ymin_2d);
            borderBufRecv_ymax_2d_host = Kokkos::create_mirror_view(borderBufRecv_ymax_2d);

        } else {
            borderBufSend_xmin_3d = FArray3d("borderBufSend_xmin", gw, jsize, ksize, npop);
            borderBufSend_xmax_3d = FArray3d("borderBufSend_xmax", gw, jsize, ksize, npop);
            borderBufSend_ymin_3d = FArray3d("borderBufSend_ymin", isize, gw, ksize, npop);
            borderBufSend_ymax_3d = FArray3d("borderBufSend_ymax", isize, gw, ksize, npop);
            borderBufSend_zmin_3d = FArray3d("borderBufSend_zmin", isize, jsize, gw, npop);
            borderBufSend_zmax_3d = FArray3d("borderBufSend_zmax", isize, jsize, gw, npop);

            borderBufRecv_xmin_3d = FArray3d("borderBufRecv_xmin", gw, jsize, ksize, npop);
            borderBufRecv_xmax_3d = FArray3d("borderBufRecv_xmax", gw, jsize, ksize, npop);
            borderBufRecv_ymin_3d = FArray3d("borderBufRecv_ymin", isize, gw, ksize, npop);
            borderBufRecv_ymax_3d = FArray3d("borderBufRecv_ymax", isize, gw, ksize, npop);
            borderBufRecv_zmin_3d = FArray3d("borderBufRecv_zmin", isize, jsize, gw, npop);
            borderBufRecv_zmax_3d = FArray3d("borderBufRecv_zmax", isize, jsize, gw, npop);

            borderBufSend_xmin_3d_host = Kokkos::create_mirror_view(borderBufSend_xmin_3d);
            borderBufSend_xmax_3d_host = Kokkos::create_mirror_view(borderBufRecv_xmax_3d);
            borderBufRecv_xmin_3d_host = Kokkos::create_mirror_view(borderBufSend_xmin_3d);
            borderBufRecv_xmax_3d_host = Kokkos::create_mirror_view(borderBufRecv_xmax_3d);

            borderBufSend_ymin_3d_host = Kokkos::create_mirror_view(borderBufSend_ymin_3d);
            borderBufSend_ymax_3d_host = Kokkos::create_mirror_view(borderBufRecv_ymax_3d);
            borderBufRecv_ymin_3d_host = Kokkos::create_mirror_view(borderBufSend_ymin_3d);
            borderBufRecv_ymax_3d_host = Kokkos::create_mirror_view(borderBufRecv_ymax_3d);

            borderBufSend_zmin_3d_host = Kokkos::create_mirror_view(borderBufSend_zmin_3d);
            borderBufSend_zmax_3d_host = Kokkos::create_mirror_view(borderBufRecv_zmax_3d);
            borderBufRecv_zmin_3d_host = Kokkos::create_mirror_view(borderBufSend_zmin_3d);
            borderBufRecv_zmax_3d_host = Kokkos::create_mirror_view(borderBufRecv_zmax_3d);
        }
#endif // USE_MPI

    }; // Periodic manager constructor



// resize last dimension of buffers
#ifdef USE_MPI
void resize_buffers(int size) {
        const int isize = params.isize;
        const int jsize = params.jsize;
        const int ksize = params.ksize;
        const int gw = params.ghostWidth;

        if (dim == 2) {
			Kokkos::realloc(borderBufSend_xmin_2d, gw, jsize, size);
			Kokkos::realloc(borderBufSend_xmax_2d, gw, jsize, size);
			Kokkos::realloc(borderBufSend_ymin_2d, isize, gw, size);
			Kokkos::realloc(borderBufSend_ymax_2d, isize, gw, size);
			
			Kokkos::realloc(borderBufRecv_xmin_2d, gw, jsize, size);
			Kokkos::realloc(borderBufRecv_xmax_2d, gw, jsize, size);
			Kokkos::realloc(borderBufRecv_ymin_2d, isize, gw, size);
			Kokkos::realloc(borderBufRecv_ymax_2d, isize, gw, size);

        } else {
			Kokkos::realloc(borderBufSend_xmin_3d, gw, jsize, ksize, size);
			Kokkos::realloc(borderBufSend_xmax_3d, gw, jsize, ksize, size);
			Kokkos::realloc(borderBufSend_ymin_3d, isize, gw, ksize, size);
			Kokkos::realloc(borderBufSend_ymax_3d, isize, gw, ksize, size);
			Kokkos::realloc(borderBufSend_zmin_3d, isize, jsize, gw, size);
			Kokkos::realloc(borderBufSend_zmax_3d, isize, jsize, gw, size);

			Kokkos::realloc(borderBufRecv_xmin_3d, gw, jsize, ksize, size);
			Kokkos::realloc(borderBufRecv_xmax_3d, gw, jsize, ksize, size);
			Kokkos::realloc(borderBufRecv_ymin_3d, isize, gw, ksize, size);
			Kokkos::realloc(borderBufRecv_ymax_3d, isize, gw, ksize, size);
			Kokkos::realloc(borderBufRecv_zmin_3d, isize, jsize, gw, size);
			Kokkos::realloc(borderBufRecv_zmax_3d, isize, jsize, gw, size);

        }
        
	}
#endif // USE_MPI



    void print_timers()
    {

        std::string s = "Rank %d TIMERS: \n";
        s += "    TOTAL     : %g\n";
        s += "    COPY      : %g\n";
        s += "    TRANSFERT : %g\n";
        s += "    COPYBACK  : %g\n";
        printf(s.c_str(), params.myRank,
            timers[TIMER_TOTAL_PERIODIC]->elapsed(),
            timers[TIMER_COPY]->elapsed(),
            timers[TIMER_TRANSFERT]->elapsed(),
            timers[TIMER_COPY_BACK]->elapsed());
    }

    void reset_timers()
    {
        timers[TIMER_TOTAL_PERIODIC]->reset();
        timers[TIMER_COPY]->reset();
        timers[TIMER_TRANSFERT]->reset();
        timers[TIMER_COPY_BACK]->reset();
    }

    // =======================================================
    // =======================================================
    /** Wrapper around make_boundaries_no_mpi / mpi */
    void make_boundaries(FArray2d fdata, BoundaryField FieldId)
    {
#ifdef USE_MPI
        make_boundaries_mpi(fdata);
        //~ printf(" mpi periodic boundary for :%d\n", FieldId);
#else
        make_boundaries_no_mpi(fdata, FieldId);
        //~ printf("no mpi periodic boundary for :%d\n", FieldId);
#endif // USE_MPI
    }

    void make_boundaries(FArray3d fdata, BoundaryField FieldId)
    {
#ifdef USE_MPI
        make_boundaries_mpi(fdata);
#else
        make_boundaries_no_mpi(fdata, FieldId);
#endif // USE_MPI
    }

    // =======================================================
    // =======================================================
    /** Apply boundary condition (no MPI) to all faces / borders.*/
    void make_boundaries_no_mpi(FArray2d fdata, BoundaryField FieldId)
    {
        // call device functors
        make_boundary_no_mpi<FACE_XMIN>(fdata, FieldId);
        make_boundary_no_mpi<FACE_XMAX>(fdata, FieldId);
        make_boundary_no_mpi<FACE_YMIN>(fdata, FieldId);
        make_boundary_no_mpi<FACE_YMAX>(fdata, FieldId);
    }
    void make_boundaries_no_mpi(FArray3d fdata, BoundaryField FieldId)
    {
        // call device functors
        make_boundary_no_mpi<FACE_XMIN>(fdata, FieldId);
        make_boundary_no_mpi<FACE_XMAX>(fdata, FieldId);
        make_boundary_no_mpi<FACE_YMIN>(fdata, FieldId);
        make_boundary_no_mpi<FACE_YMAX>(fdata, FieldId);
        make_boundary_no_mpi<FACE_ZMIN>(fdata, FieldId);
        make_boundary_no_mpi<FACE_ZMAX>(fdata, FieldId);
    }

    // =======================================================
    // =======================================================
    /** Apply boundary condition (no MPI) to a given face / border. */
    template <FaceIdType faceId>
    void make_boundary_no_mpi(FArray2d fdata, BoundaryField FieldId)
    {

        BoundaryConditionType bc_type = params.boundary_types[FieldId][faceId];

        if (bc_type == BC_PERIODIC || bc_type == BC_COPY)
            MakeBoundariesFunctor2d_Periodic<npop, faceId>::apply(params, fdata);

    } // LBMRun<dim,npop>::make_boundary - 2d

    // =======================================================
    // =======================================================
    template <FaceIdType faceId>
    void make_boundary_no_mpi(FArray3d fdata, BoundaryField FieldId)
    {

        BoundaryConditionType bc_type = params.boundary_types[FieldId][faceId];

        if (bc_type == BC_PERIODIC)
            MakeBoundariesFunctor3d_Periodic<npop, faceId>::apply(params, fdata);

    } // LBMRun<dim,npop>::make_boundary - 3d

#ifdef USE_MPI
    // =======================================================
    // =======================================================
    /** Apply boundary condition (with MPI) to all faces / borders. */

    // 2d
    void make_boundaries_mpi(FArray2d fdata)
    {

        using namespace hydroSimu;

        // for each direction:
        // 1. copy boundary to MPI buffer
        // 2. send/recv buffer
        // 3. test if BC is BC_PERIODIC / BC_COPY then ... else ..

        timers[TIMER_TOTAL_PERIODIC]->start();

        // ======
        // XDIR
        // ======
        //~ printf("Rank %d, size %d, copy x : \n", params.myRank, npop);
        copy_boundaries(fdata, XDIR);
        //~ printf("Rank %d, size %d, transfert x : \n", params.myRank, npop);
        transfert_boundaries_2d(XDIR);

        if (params.neighborsBC[X_MIN]) {
			//~ printf("Rank %d, size %d, copy from xmin : \n", params.myRank, npop);
            copy_boundaries_back(fdata, XMIN);
            
        }

        if (params.neighborsBC[X_MAX]) {
			//~ printf("Rank %d, size %d, copy from xmax : \n", params.myRank, npop);
            copy_boundaries_back(fdata, XMAX);
            
        }

        params.communicator->synchronize();

        // ======
        // YDIR
        // ======
        //~ printf("Rank %d, size %d, copy y : \n", params.myRank, npop);
        copy_boundaries(fdata, YDIR);
        //~ printf("Rank %d, size %d, transfert y : \n", params.myRank, npop);
        transfert_boundaries_2d(YDIR);

        if (params.neighborsBC[Y_MIN]) {
			//~ printf("Rank %d, size %d, copy from ymin : \n", params.myRank, npop);
            copy_boundaries_back(fdata, YMIN);
            //~ printf("Rank %d, copy to ymin side: \n", params.myRank);
        }

        if (params.neighborsBC[Y_MAX]) {
			//~ printf("Rank %d, size %d, copy from ymax : \n", params.myRank, npop);
            copy_boundaries_back(fdata, YMAX);
            //~ printf("Rank %d, copy to ymax side: \n", params.myRank);
        }

        params.communicator->synchronize();

		//~ printf("Rank %d, size %d, end make_boundaries_mpi \n", params.myRank, npop);
        timers[TIMER_TOTAL_PERIODIC]->stop();

    } // LBMRun::make_boundaries_mpi - 2d;

    // 3d
    void make_boundaries_mpi(FArray3d fdata)
    {

        using namespace hydroSimu;

        timers[TIMER_TOTAL_PERIODIC]->start();

        // ======
        // XDIR
        // ======
        copy_boundaries(fdata, XDIR);
        transfert_boundaries_3d(XDIR);

        if (params.neighborsBC[X_MIN]) {
            copy_boundaries_back(fdata, XMIN);
        }

        if (params.neighborsBC[X_MAX]) {
            copy_boundaries_back(fdata, XMAX);
        }

        params.communicator->synchronize();

        // ======
        // YDIR
        // ======
        copy_boundaries(fdata, YDIR);
        transfert_boundaries_3d(YDIR);

        if (params.neighborsBC[Y_MIN]) {
            copy_boundaries_back(fdata, YMIN);
            //~ printf("Rank %d, copy to ymin side: \n", params.myRank);
        }

        if (params.neighborsBC[Y_MAX]) {
            copy_boundaries_back(fdata, YMAX);
            //~ printf("Rank %d, copy to ymax side: \n", params.myRank);
        }

        params.communicator->synchronize();

        // ======
        // ZDIR
        // ======
        copy_boundaries(fdata, ZDIR);
        transfert_boundaries_3d(ZDIR);

        if (params.neighborsBC[Z_MIN]) {
            copy_boundaries_back(fdata, ZMIN);
        }

        if (params.neighborsBC[Z_MAX]) {
            copy_boundaries_back(fdata, ZMAX);
        }

        params.communicator->synchronize();

        timers[TIMER_TOTAL_PERIODIC]->stop();

    } // LBMRun::make_boundaries_mpi - 3d;

    /**
	 * copy boundary data into a temporary buffer for MPI communication.
	 */
    void copy_boundaries(FArray2d fdata, Direction dir)
    {

        const int isize = params.isize;
        const int jsize = params.jsize;
        const int gw = params.ghostWidth;

        timers[TIMER_COPY]->start();

        if (dir == XDIR) {

            const int nbIter = gw * jsize;
            CopyFArray_To_BorderBuf<XMIN, TWO_D>::apply(borderBufSend_xmin_2d, fdata, gw, nbIter);
            CopyFArray_To_BorderBuf<XMAX, TWO_D>::apply(borderBufSend_xmax_2d, fdata, gw, nbIter);

        }

        else if (dir == YDIR) {
            //~ printf("Rank %d, copy ydir: \n", params.myRank);
            const int nbIter = isize * gw;

            CopyFArray_To_BorderBuf<YMIN, TWO_D>::apply(borderBufSend_ymin_2d, fdata, gw, nbIter);
            CopyFArray_To_BorderBuf<YMAX, TWO_D>::apply(borderBufSend_ymax_2d, fdata, gw, nbIter);
        }

        Kokkos::fence();

        timers[TIMER_COPY]->stop();

    } // LBMRun::copy_boundaries - 2d;

    void copy_boundaries(FArray3d fdata, Direction dir)
    {

        const int isize = params.isize;
        const int jsize = params.jsize;
        const int ksize = params.ksize;
        const int gw = params.ghostWidth;

        timers[TIMER_COPY]->start();

        if (dir == XDIR) {

            const int nbIter = gw * jsize * ksize;

            CopyFArray_To_BorderBuf<XMIN, THREE_D>::apply(borderBufSend_xmin_3d, fdata, gw, nbIter);
            CopyFArray_To_BorderBuf<XMAX, THREE_D>::apply(borderBufSend_xmax_3d, fdata, gw, nbIter);
        }

        else if (dir == YDIR) {

            const int nbIter = isize * gw * ksize;

            CopyFArray_To_BorderBuf<YMIN, THREE_D>::apply(borderBufSend_ymin_3d, fdata, gw, nbIter);
            CopyFArray_To_BorderBuf<YMAX, THREE_D>::apply(borderBufSend_ymax_3d, fdata, gw, nbIter);

        }

        else if (dir == ZDIR) {

            const int nbIter = isize * jsize * gw;

            CopyFArray_To_BorderBuf<ZMIN, THREE_D>::apply(borderBufSend_zmin_3d, fdata, gw, nbIter);
            CopyFArray_To_BorderBuf<ZMAX, THREE_D>::apply(borderBufSend_zmax_3d, fdata, gw, nbIter);
        }

        Kokkos::fence();

        timers[TIMER_COPY]->stop();

    } // LBMRun::copy_boundaries - 3d;

    /**
	 * Actually perform MPI communications.
	 */
    void transfert_boundaries_2d(Direction dir)
    {

        const int data_type = params.data_type;

        using namespace hydroSimu;

        /*
	 * use MPI_Sendrecv
	 */

        // two borders to send, two borders to receive

        timers[TIMER_TRANSFERT]->start();
        MPI_Comm_rank(MPI_COMM_WORLD, &params.myRank);
        int myRank = params.myRank;
       if (dir == XDIR) {
            if (params.neighborsRank[X_MIN] != myRank || params.neighborsRank[X_MAX] != myRank) {
                // Copier une seule fois les données device -> host
                Kokkos::deep_copy(borderBufSend_xmin_2d_host, borderBufSend_xmin_2d);
                Kokkos::deep_copy(borderBufSend_xmax_2d_host, borderBufSend_xmax_2d);
            }

            // XMIN -> XMAX
            if (params.neighborsRank[X_MIN] == myRank) {
                // Intra-nœud / même GPU ou mémoire partagée
                Kokkos::deep_copy(borderBufRecv_xmax_2d, borderBufSend_xmin_2d);
            } else {
                // Inter-nœud
                params.communicator->sendrecv(borderBufSend_xmin_2d_host.data(), borderBufSend_xmin_2d_host.size(),
                    data_type, params.neighborsRank[X_MIN], 111,
                    borderBufRecv_xmax_2d_host.data(), borderBufRecv_xmax_2d_host.size(),
                    data_type, params.neighborsRank[X_MAX], 111);
                Kokkos::deep_copy(borderBufRecv_xmax_2d, borderBufRecv_xmax_2d_host);
            }

            // XMAX -> XMIN
            if (params.neighborsRank[X_MAX] == myRank) {
                Kokkos::deep_copy(borderBufRecv_xmin_2d, borderBufSend_xmax_2d);
            } else {
                params.communicator->sendrecv(borderBufSend_xmax_2d_host.data(), borderBufSend_xmax_2d_host.size(),
                    data_type, params.neighborsRank[X_MAX], 111,
                    borderBufRecv_xmin_2d_host.data(), borderBufRecv_xmin_2d_host.size(),
                    data_type, params.neighborsRank[X_MIN], 111);
                Kokkos::deep_copy(borderBufRecv_xmin_2d, borderBufRecv_xmin_2d_host);
            }
        

        } else if (dir == YDIR) {
    // Si besoin d'envoyer à un autre processus, on copie device -> host
                if (params.neighborsRank[Y_MIN] != myRank || params.neighborsRank[Y_MAX] != myRank) {
                    Kokkos::deep_copy(borderBufSend_ymin_2d_host, borderBufSend_ymin_2d);
                    Kokkos::deep_copy(borderBufSend_ymax_2d_host, borderBufSend_ymax_2d);
                }

                // YMIN -> YMAX
                if (params.neighborsRank[Y_MIN] == myRank) {
                    // Intra-nœud / même processus : copie directe sur device
                    Kokkos::deep_copy(borderBufRecv_ymax_2d, borderBufSend_ymin_2d);
                } else {
                    // Inter-nœud : échange via host
                    params.communicator->sendrecv(
                        borderBufSend_ymin_2d_host.data(), borderBufSend_ymin_2d_host.size(),
                        data_type, params.neighborsRank[Y_MIN], 211,
                        borderBufRecv_ymax_2d_host.data(), borderBufRecv_ymax_2d_host.size(),
                        data_type, params.neighborsRank[Y_MAX], 211);
                    Kokkos::deep_copy(borderBufRecv_ymax_2d, borderBufRecv_ymax_2d_host);
                }

                // YMAX -> YMIN
                if (params.neighborsRank[Y_MAX] == myRank) {
                    Kokkos::deep_copy(borderBufRecv_ymin_2d, borderBufSend_ymax_2d);
                } else {
                    params.communicator->sendrecv(
                        borderBufSend_ymax_2d_host.data(), borderBufSend_ymax_2d_host.size(),
                        data_type, params.neighborsRank[Y_MAX], 211,
                        borderBufRecv_ymin_2d_host.data(), borderBufRecv_ymin_2d_host.size(),
                        data_type, params.neighborsRank[Y_MIN], 211);
                    Kokkos::deep_copy(borderBufRecv_ymin_2d, borderBufRecv_ymin_2d_host);
                }

        }

        timers[TIMER_TRANSFERT]->stop();

    } // LBMRun<dim,npop>::transfert_boundaries_2d;
    void transfert_boundaries_3d(Direction dir)
    {

        const int data_type = params.data_type;

        using namespace hydroSimu;

        timers[TIMER_TRANSFERT]->start();
        MPI_Comm_rank(MPI_COMM_WORLD, &params.myRank);
        int myRank = params.myRank;
       if (dir == XDIR) {
            if (params.neighborsRank[X_MIN] != myRank || params.neighborsRank[X_MAX] != myRank) {
                // Copier une seule fois les données device -> host
                Kokkos::deep_copy(borderBufSend_xmin_3d_host, borderBufSend_xmin_3d);
                Kokkos::deep_copy(borderBufSend_xmax_3d_host, borderBufSend_xmax_3d);
            }

            // XMIN -> XMAX
            if (params.neighborsRank[X_MIN] == myRank) {
                // Intra-nœud / même GPU ou mémoire partagée
                Kokkos::deep_copy(borderBufRecv_xmax_3d, borderBufSend_xmin_3d);
            } else {
                // Inter-nœud
                params.communicator->sendrecv(borderBufSend_xmin_3d_host.data(), borderBufSend_xmin_3d_host.size(),
                    data_type, params.neighborsRank[X_MIN], 111,
                    borderBufRecv_xmax_3d_host.data(), borderBufRecv_xmax_3d_host.size(),
                    data_type, params.neighborsRank[X_MAX], 111);
                Kokkos::deep_copy(borderBufRecv_xmax_3d, borderBufRecv_xmax_3d_host);
            }

            // XMAX -> XMIN
            if (params.neighborsRank[X_MAX] == myRank) {
                Kokkos::deep_copy(borderBufRecv_xmin_3d, borderBufSend_xmax_3d);
            } else {
                params.communicator->sendrecv(borderBufSend_xmax_3d_host.data(), borderBufSend_xmax_3d_host.size(),
                    data_type, params.neighborsRank[X_MAX], 111,
                    borderBufRecv_xmin_3d_host.data(), borderBufRecv_xmin_3d_host.size(),
                    data_type, params.neighborsRank[X_MIN], 111);
                Kokkos::deep_copy(borderBufRecv_xmin_3d, borderBufRecv_xmin_3d_host);
            }
        


        } else if (dir == YDIR) {

                   // Si besoin d'envoyer à un autre processus, on copie device -> host
                if (params.neighborsRank[Y_MIN] != myRank || params.neighborsRank[Y_MAX] != myRank) {
                    Kokkos::deep_copy(borderBufSend_ymin_3d_host, borderBufSend_ymin_3d);
                    Kokkos::deep_copy(borderBufSend_ymax_3d_host, borderBufSend_ymax_3d);
                }

                // YMIN -> YMAX
                if (params.neighborsRank[Y_MIN] == myRank) {
                    // Intra-nœud / même processus : copie directe sur device
                    Kokkos::deep_copy(borderBufRecv_ymax_3d, borderBufSend_ymin_3d);
                } else {
                    // Inter-nœud : échange via host
                    params.communicator->sendrecv(
                        borderBufSend_ymin_3d_host.data(), borderBufSend_ymin_3d_host.size(),
                        data_type, params.neighborsRank[Y_MIN], 211,
                        borderBufRecv_ymax_3d_host.data(), borderBufRecv_ymax_3d_host.size(),
                        data_type, params.neighborsRank[Y_MAX], 211);
                    Kokkos::deep_copy(borderBufRecv_ymax_3d, borderBufRecv_ymax_3d_host);
                }

                // YMAX -> YMIN
                if (params.neighborsRank[Y_MAX] == myRank) {
                    Kokkos::deep_copy(borderBufRecv_ymin_3d, borderBufSend_ymax_3d);
                } else {
                    params.communicator->sendrecv(
                        borderBufSend_ymax_3d_host.data(), borderBufSend_ymax_3d_host.size(),
                        data_type, params.neighborsRank[Y_MAX], 211,
                        borderBufRecv_ymin_3d_host.data(), borderBufRecv_ymin_3d_host.size(),
                        data_type, params.neighborsRank[Y_MIN], 211);
                    Kokkos::deep_copy(borderBufRecv_ymin_3d, borderBufRecv_ymin_3d_host);
                }

        } else if (dir == ZDIR) {

                 // Copier device -> host uniquement si c’est inter-processus
              if (params.neighborsRank[Z_MIN] != myRank || params.neighborsRank[Z_MAX] != myRank) {
                Kokkos::deep_copy(borderBufSend_zmin_3d_host, borderBufSend_zmin_3d);
                Kokkos::deep_copy(borderBufSend_zmax_3d_host, borderBufSend_zmax_3d);
            }

            // ZMIN -> ZMAX
            if (params.neighborsRank[Z_MIN] == myRank) {
                Kokkos::deep_copy(borderBufRecv_zmax_3d, borderBufSend_zmin_3d);
            } else {
                params.communicator->sendrecv(
                    borderBufSend_zmin_3d_host.data(), borderBufSend_zmin_3d_host.size(),
                    data_type, params.neighborsRank[Z_MIN], 311,
                    borderBufRecv_zmax_3d_host.data(), borderBufRecv_zmax_3d_host.size(),
                    data_type, params.neighborsRank[Z_MAX], 311);
                Kokkos::deep_copy(borderBufRecv_zmax_3d, borderBufRecv_zmax_3d_host);
            }

            // ZMAX -> ZMIN
            if (params.neighborsRank[Z_MAX] == myRank) {
                Kokkos::deep_copy(borderBufRecv_zmin_3d, borderBufSend_zmax_3d);
            } else {
                params.communicator->sendrecv(
                    borderBufSend_zmax_3d_host.data(), borderBufSend_zmax_3d_host.size(),
                    data_type, params.neighborsRank[Z_MAX], 311,
                    borderBufRecv_zmin_3d_host.data(), borderBufRecv_zmin_3d_host.size(),
                    data_type, params.neighborsRank[Z_MIN], 311);
                Kokkos::deep_copy(borderBufRecv_zmin_3d, borderBufRecv_zmin_3d_host);
            }
        }

        timers[TIMER_TRANSFERT]->stop();

    } // LBMRun<dim,npop>::transfert_boundaries_3d;

    /**
	 * copy temporary buffer (from MPI communication) to boundary data.
	 */
    void copy_boundaries_back(FArray2d fdata, BoundaryLocation loc)
    {

        const int isize = params.isize;
        const int jsize = params.jsize;
        //const int ksize = params.ksize;
        const int gw = params.ghostWidth;

        timers[TIMER_COPY_BACK]->start();

        if (loc == XMIN) {

            const int nbIter = gw * jsize;

            CopyBorderBuf_To_FArray<XMIN, TWO_D>::apply(fdata, borderBufRecv_xmin_2d, gw, nbIter);
        }

        if (loc == XMAX) {

            const int nbIter = gw * jsize;

            CopyBorderBuf_To_FArray<XMAX, TWO_D>::apply(fdata, borderBufRecv_xmax_2d, gw, nbIter);
        }

        if (loc == YMIN) {

            const int nbIter = isize * gw;

            CopyBorderBuf_To_FArray<YMIN, TWO_D>::apply(fdata, borderBufRecv_ymin_2d, gw, nbIter);
        }

        if (loc == YMAX) {

            const int nbIter = isize * gw;

            CopyBorderBuf_To_FArray<YMAX, TWO_D>::apply(fdata, borderBufRecv_ymax_2d, gw, nbIter);
        }

        timers[TIMER_COPY_BACK]->stop();

    } // LBMRun<dim,npop>::copy_boundaries_back - 2d

    void copy_boundaries_back(FArray3d fdata, BoundaryLocation loc)
    {

        const int isize = params.isize;
        const int jsize = params.jsize;
        const int ksize = params.ksize;
        const int gw = params.ghostWidth;

        timers[TIMER_COPY_BACK]->start();

        if (loc == XMIN) {

            const int nbIter = gw * jsize * ksize;

            CopyBorderBuf_To_FArray<XMIN, THREE_D>::apply(fdata, borderBufRecv_xmin_3d, gw, nbIter);
        }

        if (loc == XMAX) {

            const int nbIter = gw * jsize * ksize;

            CopyBorderBuf_To_FArray<XMAX, THREE_D>::apply(fdata, borderBufRecv_xmax_3d, gw, nbIter);
        }

        if (loc == YMIN) {

            const int nbIter = isize * gw * ksize;

            CopyBorderBuf_To_FArray<YMIN, THREE_D>::apply(fdata, borderBufRecv_ymin_3d, gw, nbIter);
        }

        if (loc == YMAX) {

            const int nbIter = isize * gw * ksize;

            CopyBorderBuf_To_FArray<YMAX, THREE_D>::apply(fdata, borderBufRecv_ymax_3d, gw, nbIter);
        }

        if (loc == ZMIN) {

            const int nbIter = isize * jsize * gw;

            CopyBorderBuf_To_FArray<ZMIN, THREE_D>::apply(fdata, borderBufRecv_zmin_3d, gw, nbIter);
        }

        if (loc == ZMAX) {

            const int nbIter = isize * jsize * gw;

            CopyBorderBuf_To_FArray<ZMAX, THREE_D>::apply(fdata, borderBufRecv_zmax_3d, gw, nbIter);
        }

        timers[TIMER_COPY_BACK]->stop();

    } // LBMRun<dim,npop>::copy_boundaries_back - 3d;

#endif // USE_MPI

protected:
#ifdef USE_MPI
    //! \defgroup BorderBuffer data arrays for border exchange handling
    //! we assume that we use a cuda-aware version of OpenMPI / MVAPICH
    //! @{
    FArray2d borderBufSend_xmin_2d;
    FArray2d borderBufSend_xmax_2d;
    FArray2d borderBufSend_ymin_2d;
    FArray2d borderBufSend_ymax_2d;

    FArray2d borderBufRecv_xmin_2d;
    FArray2d borderBufRecv_xmax_2d;
    FArray2d borderBufRecv_ymin_2d;
    FArray2d borderBufRecv_ymax_2d;


    FArray2dHost borderBufSend_xmin_2d_host;
    FArray2dHost borderBufSend_xmax_2d_host;
    FArray2dHost borderBufRecv_xmin_2d_host;
    FArray2dHost borderBufRecv_xmax_2d_host;

    FArray2dHost borderBufSend_ymin_2d_host;
    FArray2dHost borderBufSend_ymax_2d_host;
    FArray2dHost borderBufRecv_ymin_2d_host;
    FArray2dHost borderBufRecv_ymax_2d_host;

    FArray3d borderBufSend_xmin_3d;
    FArray3d borderBufSend_xmax_3d;
    FArray3d borderBufSend_ymin_3d;
    FArray3d borderBufSend_ymax_3d;
    FArray3d borderBufSend_zmin_3d;
    FArray3d borderBufSend_zmax_3d;

    FArray3d borderBufRecv_xmin_3d;
    FArray3d borderBufRecv_xmax_3d;
    FArray3d borderBufRecv_ymin_3d;
    FArray3d borderBufRecv_ymax_3d;
    FArray3d borderBufRecv_zmin_3d;
    FArray3d borderBufRecv_zmax_3d;

    FArray3dHost borderBufSend_xmin_3d_host;
    FArray3dHost borderBufSend_xmax_3d_host;
    FArray3dHost borderBufRecv_xmin_3d_host;
    FArray3dHost borderBufRecv_xmax_3d_host;

    FArray3dHost borderBufSend_ymin_3d_host;
    FArray3dHost borderBufSend_ymax_3d_host;
    FArray3dHost borderBufRecv_ymin_3d_host;
    FArray3dHost borderBufRecv_ymax_3d_host;

    FArray3dHost borderBufSend_zmin_3d_host;
    FArray3dHost borderBufSend_zmax_3d_host;
    FArray3dHost borderBufRecv_zmin_3d_host;
    FArray3dHost borderBufRecv_zmax_3d_host;
    //! @}
#endif // USE_MPI

}; // end class FArrayManager

#endif // FARRAY_MANAGER_H_
