#ifndef LBM_SCHEME_BASE_H_
#define LBM_SCHEME_BASE_H_

#include <Kokkos_Random.hpp>
#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>

#endif // __CUDA_ARCH__

#include "FieldManager.h"
#include "LBMParams.h"
#include "LBM_Base_Functor.h"

template<int dim, int npop>
struct LBMSchemeBase
{

    static constexpr int tp_dim = dim;

    using LBMArray = typename LBMBaseFunctor<dim, npop>::LBMArray;
    using LBMArrayHost = typename LBMBaseFunctor<dim, npop>::LBMArrayHost;
    using FArray = typename LBMBaseFunctor<dim, npop>::FArray;
    using LBM_Weights = typename LBMBaseFunctor<dim, npop>::LBM_Weights;
    using LBM_speeds = typename LBMBaseFunctor<dim, npop>::LBM_speeds;
    using LBM_speeds_opposite = typename LBMBaseFunctor<dim, npop>::LBM_speeds_opposite;
    using FState = typename Kokkos::Array<real_t, npop>;

    LBMParams params;
    LBMArray lbm_data;

    FArray f1, f2, f3, f4, f5;
    FArray f_tmp;

    LBMArray CCdata;
    uset map_CCid;
    umap map_CCid_to_rank;

    LBMLattice<dim, npop> lattice;
    LBM_speeds E;
    LBM_speeds_opposite Ebar;
    LBM_Weights w;

    int nbVar;
    LBMSchemeBase()
      : lattice(LBMLattice<dim, npop>())
      , E(LBMLattice<dim, npop>().E)
      , Ebar(LBMLattice<dim, npop>().Ebar)
      , w(LBMLattice<dim, npop>().w){};
    // constructor for 2 equation (ie 2 lbm distribution of type FArray)
    LBMSchemeBase(const LBMParams& params, LBMArray lbm_data)
      : params(params)
      , lbm_data(lbm_data)
      , map_CCid(KOKKOS_MAP_DEFAULT_CAPACITY)
      , map_CCid_to_rank(KOKKOS_MAP_DEFAULT_CAPACITY)
      , lattice(LBMLattice<dim, npop>())
      , E(LBMLattice<dim, npop>().E)
      , Ebar(LBMLattice<dim, npop>().Ebar)
      , w(LBMLattice<dim, npop>().w)
    {
        
    };
    
    void allocate_f(int nbEqs){
        if (dim == 2)
        {
            nbVar = lbm_data.extent(2);
        }
        else if (dim == 3)
        {
            nbVar = lbm_data.extent(3);
        }

        const int isize = params.isize;
        const int jsize = params.jsize;
        const int ksize = params.ksize;
        // memory allocation (use sizes with ghosts included)
        if (dim == 2)
        {
            if (nbEqs >= 1)
            {
                f1 = FArray("f1", isize, jsize, npop);
            }
            if (nbEqs >= 2)
            {
                f2 = FArray("f2", isize, jsize, npop);
            }
            if (nbEqs >= 3)
            {
                f3 = FArray("f3", isize, jsize, npop);
            }
            if (nbEqs >= 4)
            {
                f4 = FArray("f4", isize, jsize, npop);
            }
            if (nbEqs >= 5)
            {
                f5 = FArray("f5", isize, jsize, npop);
            }

            f_tmp = FArray("f_tmp", isize, jsize, npop);
            CCdata = LBMArray("cc_data", isize, jsize, 1);
        }
        else if (dim == 3)
        {
            if (nbEqs >= 1)
            {
                f1 = FArray("f1", isize, jsize, ksize, npop);
            }
            if (nbEqs >= 2)
            {
                f2 = FArray("f2", isize, jsize, ksize, npop);
            }
            if (nbEqs >= 3)
            {
                f3 = FArray("f3", isize, jsize, ksize, npop);
            }
            if (nbEqs >= 4)
            {
                f4 = FArray("f4", isize, jsize, ksize, npop);
            }
            if (nbEqs >= 5)
            {
                f5 = FArray("f5", isize, jsize, ksize, npop);
            }

            f_tmp = FArray("f_tmp", isize, jsize, ksize, npop);
            CCdata = LBMArray("cc_data", isize, jsize, ksize, 1);
        }
    };

    // ================================================
    //  swap distribution with f_tmp containing new distribution after collide and
    //  stream
    // ================================================
    void swap_distribution(EquationTag1)
    {
        FArray tmp = f1;
        f1 = f_tmp;
        f_tmp = tmp;
    }
    void swap_distribution(EquationTag2)
    {
        FArray tmp = f2;
        f2 = f_tmp;
        f_tmp = tmp;
    }
    void swap_distribution(EquationTag3)
    {
        FArray tmp = f3;
        f3 = f_tmp;
        f_tmp = tmp;
    }
    void swap_distribution(EquationTag4)
    {
        FArray tmp = f4;
        f4 = f_tmp;
        f_tmp = tmp;
    }
    void swap_distribution(EquationTag5)
    {
        FArray tmp = f5;
        f5 = f_tmp;
        f_tmp = tmp;
    }

    // ================================================
    //  manage array values with coordinates vector templated on dim
    // ================================================
    KOKKOS_INLINE_FUNCTION real_t get_lbm_val(const IVect2 ijk,
                                              int ifield) const
    {
        return lbm_data(ijk[IX], ijk[IY], ifield);
    }
    KOKKOS_INLINE_FUNCTION real_t get_lbm_val(const IVect3 ijk,
                                              int ifield) const
    {
        return lbm_data(ijk[IX], ijk[IY], ijk[IZ], ifield);
    }

    KOKKOS_INLINE_FUNCTION void set_lbm_val(const IVect2 ijk, int ifield, real_t val) const
    {
        lbm_data(ijk[IX], ijk[IY], ifield) = val;
    }
    KOKKOS_INLINE_FUNCTION void set_lbm_val(const IVect3 ijk, int ifield, real_t val) const
    {
        lbm_data(ijk[IX], ijk[IY], ijk[IZ], ifield) = val;
    }

    KOKKOS_INLINE_FUNCTION real_t get_ftmp_val(const IVect2 ijk, int ipop) const
    {
        return f_tmp(ijk[0], ijk[1], ipop);
    }
    KOKKOS_INLINE_FUNCTION real_t get_ftmp_val(const IVect3 ijk, int ipop) const
    {
        return f_tmp(ijk[0], ijk[1], ijk[IZ], ipop);
    }

    KOKKOS_INLINE_FUNCTION void set_ftmp_val(const IVect2 ijk, int ipop, real_t val) const
    {
        f_tmp(ijk[0], ijk[1], ipop) = val;
    }
    KOKKOS_INLINE_FUNCTION void set_ftmp_val(const IVect3 ijk, int ipop, real_t val) const
    {
        f_tmp(ijk[IX], ijk[IY], ijk[IZ], ipop) = val;
    }

    KOKKOS_INLINE_FUNCTION real_t get_f_val(EquationTag1, const IVect2 ijk, int ipop) const
    {
        return f1(ijk[0], ijk[1], ipop);
    }
    KOKKOS_INLINE_FUNCTION real_t get_f_val(EquationTag2, const IVect2 ijk, int ipop) const
    {
        return f2(ijk[0], ijk[1], ipop);
    }
    KOKKOS_INLINE_FUNCTION real_t get_f_val(EquationTag3, const IVect2 ijk, int ipop) const
    {
        return f3(ijk[0], ijk[1], ipop);
    }
    KOKKOS_INLINE_FUNCTION real_t get_f_val(EquationTag4, const IVect2 ijk, int ipop) const
    {
        return f4(ijk[0], ijk[1], ipop);
    }
    KOKKOS_INLINE_FUNCTION real_t get_f_val(EquationTag5, const IVect2 ijk, int ipop) const
    {
        return f5(ijk[0], ijk[1], ipop);
    }

    KOKKOS_INLINE_FUNCTION real_t get_f_val(EquationTag1, const IVect3 ijk, int ipop) const
    {
        return f1(ijk[0], ijk[1], ijk[2], ipop);
    }
    KOKKOS_INLINE_FUNCTION real_t get_f_val(EquationTag2, const IVect3 ijk, int ipop) const
    {
        return f2(ijk[0], ijk[1], ijk[2], ipop);
    }
    KOKKOS_INLINE_FUNCTION real_t get_f_val(EquationTag3, const IVect3 ijk, int ipop) const
    {
        return f3(ijk[0], ijk[1], ijk[2], ipop);
    }
    KOKKOS_INLINE_FUNCTION real_t get_f_val(EquationTag4, const IVect3 ijk, int ipop) const
    {
        return f4(ijk[0], ijk[1], ijk[2], ipop);
    }
    KOKKOS_INLINE_FUNCTION real_t get_f_val(EquationTag5, const IVect3 ijk, int ipop) const
    {
        return f5(ijk[0], ijk[1], ijk[2], ipop);
    }

    KOKKOS_INLINE_FUNCTION void set_f_val(EquationTag1, const IVect2 ijk, int ipop, real_t val) const
    {
        f1(ijk[0], ijk[1], ipop) = val;
    }
    KOKKOS_INLINE_FUNCTION void set_f_val(EquationTag2, const IVect2 ijk, int ipop, real_t val) const
    {
        f2(ijk[0], ijk[1], ipop) = val;
    }
    KOKKOS_INLINE_FUNCTION void set_f_val(EquationTag3, const IVect2 ijk, int ipop, real_t val) const
    {
        f3(ijk[0], ijk[1], ipop) = val;
    }
    KOKKOS_INLINE_FUNCTION void set_f_val(EquationTag4, const IVect2 ijk, int ipop, real_t val) const
    {
        f4(ijk[0], ijk[1], ipop) = val;
    }
    KOKKOS_INLINE_FUNCTION void set_f_val(EquationTag5, const IVect2 ijk, int ipop, real_t val) const
    {
        f5(ijk[0], ijk[1], ipop) = val;
    }

    KOKKOS_INLINE_FUNCTION void set_f_val(EquationTag1, const IVect3 ijk, int ipop, real_t val) const
    {
        f1(ijk[0], ijk[1], ijk[2], ipop) = val;
    }
    KOKKOS_INLINE_FUNCTION void set_f_val(EquationTag2, const IVect3 ijk, int ipop, real_t val) const
    {
        f2(ijk[0], ijk[1], ijk[2], ipop) = val;
    }
    KOKKOS_INLINE_FUNCTION void set_f_val(EquationTag3, const IVect3 ijk, int ipop, real_t val) const
    {
        f3(ijk[0], ijk[1], ijk[2], ipop) = val;
    }
    KOKKOS_INLINE_FUNCTION void set_f_val(EquationTag4, const IVect3 ijk, int ipop, real_t val) const
    {
        f4(ijk[0], ijk[1], ijk[2], ipop) = val;
    }
    KOKKOS_INLINE_FUNCTION void set_f_val(EquationTag5, const IVect3 ijk, int ipop, real_t val) const
    {
        f5(ijk[0], ijk[1], ijk[2], ipop) = val;
    }

    KOKKOS_INLINE_FUNCTION real_t get_cc_label(const IVect2& ijk) const
    {
        return CCdata(ijk[IX], ijk[IY], 0);
    }
    KOKKOS_INLINE_FUNCTION real_t get_cc_label(const IVect3& ijk) const
    {
        return CCdata(ijk[IX], ijk[IY], ijk[IZ], 0);
    }


    KOKKOS_INLINE_FUNCTION void set_cc_label(const IVect2& ijk,
                                             real_t val) const
    {
        CCdata(ijk[IX], ijk[IY], 0) = val;
    }
    KOKKOS_INLINE_FUNCTION void set_cc_label(const IVect3& ijk,
                                             real_t val) const
    {
        CCdata(ijk[IX], ijk[IY], ijk[IZ], 0) = val;
    }

    // ================================================
    //  get macro local values into a LBMState vector for use in model parameters
    //  functions
    // ================================================
    template<typename LBMState>
    KOKKOS_INLINE_FUNCTION void setupLBMState(const IVect2 IJK,
                                              LBMState& lbmState) const
    {
        const int i = IJK[IX];
        const int j = IJK[IY];
        for (int IVAR = 0; IVAR < nbVar; ++IVAR)
        {
            lbmState[IVAR] = lbm_data(i, j, IVAR);
        }
    }
    template<typename LBMState>
    KOKKOS_INLINE_FUNCTION void setupLBMState(const IVect3 IJK,
                                              LBMState& lbmState) const
    {
        const int i = IJK[IX];
        const int j = IJK[IY];
        const int k = IJK[IZ];
        for (int IVAR = 0; IVAR < nbVar; ++IVAR)
        {
            lbmState[IVAR] = lbm_data(i, j, k, IVAR);
        }
    }
    template<typename LBMState, int NBVAR>
    KOKKOS_INLINE_FUNCTION void setupLBMState2(const IVect2 IJK,
                                               LBMState& lbmState) const
    {
        const int i = IJK[IX];
        const int j = IJK[IY];
        for (int IVAR = 0; IVAR < NBVAR; ++IVAR)
        {
            lbmState[IVAR] = lbm_data(i, j, IVAR);
        }
    }
    template<typename LBMState, int NBVAR>
    KOKKOS_INLINE_FUNCTION void setupLBMState2(const IVect3 IJK,
                                               LBMState& lbmState) const
    {
        const int i = IJK[IX];
        const int j = IJK[IY];
        const int k = IJK[IZ];
        for (int IVAR = 0; IVAR < NBVAR; ++IVAR)
        {
            lbmState[IVAR] = lbm_data(i, j, k, IVAR);
        }
    }

    // ================================================
    //  use colliders to init
    // ================================================
    template<typename EquationTagTemp, typename Collider>
    KOKKOS_INLINE_FUNCTION void set_f_as_feq(EquationTagTemp tag,
                                             const IVect<dim>& IJK,
                                             const Collider& collider) const
    {
        for (int ipop = 0; ipop < npop; ++ipop)
        {
            set_f_val(tag, IJK, ipop, collider.get_feq(ipop));
        }
    }

    // ================================================
    //  manage bounds
    // ================================================
    KOKKOS_INLINE_FUNCTION int get_bound(EquationTag1, int FaceId) const
    {
        return params.bounds[BOUNDARY_EQUATION_1][FaceId];
    }
    KOKKOS_INLINE_FUNCTION int get_bound(EquationTag2, int FaceId) const
    {
        return params.bounds[BOUNDARY_EQUATION_2][FaceId];
    }
    KOKKOS_INLINE_FUNCTION int get_bound(EquationTag3, int FaceId) const
    {
        return params.bounds[BOUNDARY_EQUATION_3][FaceId];
    }
    KOKKOS_INLINE_FUNCTION int get_bound(EquationTag4, int FaceId) const
    {
        return params.bounds[BOUNDARY_EQUATION_4][FaceId];
    }
    KOKKOS_INLINE_FUNCTION int get_bound(EquationTag5, int FaceId) const
    {
        return params.bounds[BOUNDARY_EQUATION_5][FaceId];
    }

    // ================================================
    //  check if we are inside the domain to update (because ghost cells should
    //  not stream inside the domain unless the boundary is periodic)
    // ================================================
    // is_in_bounds 2d
    template<typename EquationTag>
    KOKKOS_INLINE_FUNCTION bool is_in_bounds(EquationTag tag,
                                             const IVect2& IJK) const
    {
        return (IJK[IX] >= get_bound(tag, FACE_XMIN) and
                IJK[IX] <= get_bound(tag, FACE_XMAX) and
                IJK[IY] >= get_bound(tag, FACE_YMIN) and
                IJK[IY] <= get_bound(tag, FACE_YMAX));
    }

    // is_in_bounds 3d
    template<typename EquationTag>
    KOKKOS_INLINE_FUNCTION bool is_in_bounds(EquationTag tag,
                                             const IVect3& IJK) const
    {
        return (IJK[IX] >= get_bound(tag, FACE_XMIN) and
                IJK[IX] <= get_bound(tag, FACE_XMAX) and
                IJK[IY] >= get_bound(tag, FACE_YMIN) and
                IJK[IY] <= get_bound(tag, FACE_YMAX) and
                IJK[IZ] >= get_bound(tag, FACE_ZMIN) and
                IJK[IZ] <= get_bound(tag, FACE_ZMAX));
    }

    // ================================================
    //  check if a coord is a not ghostcell
    // ================================================
    KOKKOS_INLINE_FUNCTION
    bool isNotGhostCell(const IVect2& IJK) const
    {
        int gw = params.ghostWidth;
        return (IJK[IX] > gw - 1 && IJK[IX] < params.isize - gw &&
                IJK[IY] > gw - 1 && IJK[IY] < params.jsize - gw);
    }
    KOKKOS_INLINE_FUNCTION
    bool isNotGhostCell(const IVect3& IJK) const
    {
        int gw = params.ghostWidth;
        return (IJK[IX] > gw - 1 && IJK[IX] < params.isize - gw &&
                IJK[IY] > gw - 1 && IJK[IY] < params.jsize - gw &&
                IJK[IZ] > gw - 1 && IJK[IZ] < params.ksize - gw);
    }

    bool isGhostCell(const IVect2& IJK) const
    {
        return (not(isNotGhostCell(IJK)));
    }
    bool isGhostCell(const IVect3& IJK) const
    {
        return (not(isNotGhostCell(IJK)));
    }

    // ================================================
    //  stream functions for each direction
    // ================================================

    KOKKOS_INLINE_FUNCTION bool stream_alldir(const IVect2& IJK, IVect2& IJKs, int ipop) const
    {
        const int i = IJK[IX];
        const int j = IJK[IY];
        const int ip = lattice.E[ipop][IX];
        const int jp = lattice.E[ipop][IY];
        IJKs[IX] = i + ip;
        IJKs[IY] = j + jp;
        return (IJKs[IX] >= 0 and IJKs[IX] < params.isize and IJKs[IY] >= 0 and
                IJKs[IY] < params.jsize);
    }
    KOKKOS_INLINE_FUNCTION bool stream_alldir(const IVect3& IJK, IVect3& IJKs, int ipop) const
    {
        const int i = IJK[IX];
        const int j = IJK[IY];
        const int k = IJK[IZ];
        const int ip = lattice.E[ipop][IX];
        const int jp = lattice.E[ipop][IY];
        const int kp = lattice.E[ipop][IZ];
        IJKs[IX] = i + ip;
        IJKs[IY] = j + jp;
        IJKs[IZ] = k + kp;
        return (IJKs[IX] >= 0 and IJKs[IX] < params.isize and IJKs[IY] >= 0 and
                IJKs[IY] < params.jsize and IJKs[IZ] >= 0 and
                IJKs[IZ] < params.ksize);
    }

    // ================================================
    //  check if it is possible to stream from position i, j in direction ipop
    // ================================================
    KOKKOS_INLINE_FUNCTION
    bool can_stream(IVect2 IJK, int ipop) const
    {
        return (IJK[IX] + lattice.E[ipop][IX] >= 0 and
                IJK[IX] + E[ipop][IX] < params.isize and
                IJK[IY] + lattice.E[ipop][IY] >= 0 and
                IJK[IY] + E[ipop][IY] < params.jsize);
    }
    KOKKOS_INLINE_FUNCTION
    bool can_stream(IVect3 IJK, int ipop) const
    {
        return (IJK[IX] + lattice.E[ipop][IX] >= 0 and
                IJK[IX] + lattice.E[ipop][IX] < params.isize and
                IJK[IY] + lattice.E[ipop][IY] >= 0 and
                IJK[IY] + lattice.E[ipop][IY] < params.jsize and
                IJK[IZ] + lattice.E[ipop][IZ] >= 0 and
                IJK[IZ] + lattice.E[ipop][IZ] < params.ksize);
    }

    KOKKOS_INLINE_FUNCTION
    bool can_stream2(IVect2 IJKs) const
    {
        return (IJKs[IX] >= 0 and IJKs[IX] < params.isize and IJKs[IY] >= 0 and
                IJKs[IY] < params.jsize);
    }
    KOKKOS_INLINE_FUNCTION
    bool can_stream2(IVect3 IJKs) const
    {
        return (IJKs[IX] >= 0 and IJKs[IX] < params.isize and IJKs[IY] >= 0 and
                IJKs[IY] < params.jsize and IJKs[IZ] >= 0 and
                IJKs[IZ] < params.ksize);
    }

    // ================================================
    //  use colliders to stream
    // ================================================
    // stream 2d
    template<typename Collider>
    KOKKOS_INLINE_FUNCTION void stream(IVect<dim> IJK,
                                       const Collider& collider) const
    {
        IVect<dim> IJKs;
        bool can_str;
        for (int ipop = 0; ipop < npop; ++ipop)
        {
            can_str = stream_alldir(IJK, IJKs, ipop);
            if (can_str)
            {
                set_ftmp_val(IJKs, ipop, collider.collide(ipop));
            }
        }
    }

    // ================================================
    //  compute scalar products with lattice vector ipop
    // ================================================
    // compute scal 2d
    KOKKOS_INLINE_FUNCTION
    real_t compute_scal(int ipop, real_t Vx, real_t Vy) const
    {
        return (lattice.E[ipop][IX] * Vx + lattice.E[ipop][IY] * Vy);
    }
    // compute scal 3d
    KOKKOS_INLINE_FUNCTION
    real_t compute_scal(int ipop, real_t Vx, real_t Vy, real_t Vz) const
    {
        return (lattice.E[ipop][IX] * Vx + lattice.E[ipop][IY] * Vy +
                lattice.E[ipop][IZ] * Vz);
    }
    // compute scal 2d
    KOKKOS_INLINE_FUNCTION
    real_t compute_scal(int ipop, const RVect2& V) const
    {
        return (lattice.E[ipop][IX] * V[IX] + lattice.E[ipop][IY] * V[IY]);
    }
    // compute scal 3d
    KOKKOS_INLINE_FUNCTION
    real_t compute_scal(int ipop, const RVect3& V) const
    {
        return (lattice.E[ipop][IX] * V[IX] + lattice.E[ipop][IY] * V[IY] +
                lattice.E[ipop][IZ] * V[IZ]);
    }
    // compute scal 2d
    KOKKOS_INLINE_FUNCTION
    real_t compute_scal_Ebar(int ipop, real_t Vx, real_t Vy) const
    {
        return (lattice.E[Ebar[ipop]][IX] * Vx + lattice.E[Ebar[ipop]][IY] * Vy);
    }
    // compute scal 3d
    KOKKOS_INLINE_FUNCTION
    real_t compute_scal_Ebar(int ipop, real_t Vx, real_t Vy, real_t Vz) const
    {
        return (lattice.E[lattice.Ebar[ipop]][IX] * Vx +
                lattice.E[lattice.Ebar[ipop]][IY] * Vy +
                lattice.E[lattice.Ebar[ipop]][IZ] * Vz);
    }

    // ================================================
    //  get x y z coord of a point of indices IJK
    // ================================================
    KOKKOS_INLINE_FUNCTION
    void get_coordinates(const IVect2 IJK, real_t& x, real_t& y) const
    {
        const int ghostWidth = params.ghostWidth;
        const int nx = params.nx;
        const int ny = params.ny;

        const real_t xmin = params.xmin;
        const real_t ymin = params.ymin;
        const real_t dx = params.dx;
        const real_t dy = params.dy;

// MPI domain coordinates
#ifdef USE_MPI
        const int i_mpi = params.myMpiPos[IX];
        const int j_mpi = params.myMpiPos[IY];
#else
        const int i_mpi = 0;
        const int j_mpi = 0;
#endif

        x = xmin + dx / 2 + (IJK[IX] + nx * i_mpi - ghostWidth) * dx;
        y = ymin + dy / 2 + (IJK[IY] + ny * j_mpi - ghostWidth) * dy;
    }
    KOKKOS_INLINE_FUNCTION
    void get_coordinates(const IVect3 IJK, real_t& x, real_t& y) const
    {
        const int ghostWidth = params.ghostWidth;
        const int nx = params.nx;
        const int ny = params.ny;

        const real_t xmin = params.xmin;
        const real_t ymin = params.ymin;
        const real_t dx = params.dx;
        const real_t dy = params.dy;

// MPI domain coordinates
#ifdef USE_MPI
        const int i_mpi = params.myMpiPos[IX];
        const int j_mpi = params.myMpiPos[IY];
#else
        const int i_mpi = 0;
        const int j_mpi = 0;
#endif

        x = xmin + dx / 2 + (IJK[IX] + nx * i_mpi - ghostWidth) * dx;
        y = ymin + dy / 2 + (IJK[IY] + ny * j_mpi - ghostWidth) * dy;
    }
    KOKKOS_INLINE_FUNCTION
    void get_coord3D(const IVect2 IJK, real_t& x, real_t& y, real_t& z) const
    {
        const int ghostWidth = params.ghostWidth;
        const int nx = params.nx;
        const int ny = params.ny;

        const real_t xmin = params.xmin;
        const real_t ymin = params.ymin;
        const real_t dx = params.dx;
        const real_t dy = params.dy;

// MPI domain coordinates
#ifdef USE_MPI
        const int i_mpi = params.myMpiPos[IX];
        const int j_mpi = params.myMpiPos[IY];
#else
        const int i_mpi = 0;
        const int j_mpi = 0;
#endif

        x = xmin + dx / 2 + (IJK[IX] + nx * i_mpi - ghostWidth) * dx;
        y = ymin + dy / 2 + (IJK[IY] + ny * j_mpi - ghostWidth) * dy;
    }

    KOKKOS_INLINE_FUNCTION
    void get_coord3D(const IVect3 IJK, real_t& x, real_t& y, real_t& z) const
    {
        const int ghostWidth = params.ghostWidth;
        const int nx = params.nx;
        const int ny = params.ny;
        const int nz = params.nz;

        const real_t xmin = params.xmin;
        const real_t ymin = params.ymin;
        const real_t zmin = params.zmin;
        const real_t dx = params.dx;
        const real_t dy = params.dy;
        const real_t dz = params.dz;

// MPI domain coordinates
#ifdef USE_MPI
        const int i_mpi = params.myMpiPos[IX];
        const int j_mpi = params.myMpiPos[IY];
        const int k_mpi = params.myMpiPos[IZ];
#else
        const int i_mpi = 0;
        const int j_mpi = 0;
        const int k_mpi = 0;
#endif

    x = xmin + dx / 2 + (IJK[IX] + nx * i_mpi - ghostWidth) * dx;
    y = ymin + dy / 2 + (IJK[IY] + ny * j_mpi - ghostWidth) * dy;
    z = zmin + dz / 2 + (IJK[IZ] + nz * k_mpi - ghostWidth) * dz;
  }
  // ================================================
  //  Compute gradient of ifield - 2d
  // ================================================
  KOKKOS_INLINE_FUNCTION
  void compute_gradient(RVect2 &grad, const IVect2 &IJK, int ifield,
                        int boundary) const {

    const real_t dx = params.dx;
    const real_t e2 = 3.0;

    int i = IJK[IX];
    int j = IJK[IY];

    Kokkos::Array<int, 6> bounds(params.bounds[boundary]);
    //~ int use_CDF2=0;
    //~ const bool use_offcenter = (i <= bounds[XMIN]+use_CDF2 or i >=
    //bounds[XMAX]-use_CDF2 or j <= bounds[YMIN]+use_CDF2 or j >=
    //bounds[YMAX]-use_CDF2);
    const bool use_offcenter = (i <= bounds[XMIN] or i >= bounds[XMAX] or
                                j <= bounds[YMIN] or j >= bounds[YMAX]);
    real_t dfdx = 0.0, dfdy = 0.0;
    if (use_offcenter) {
      if (i <= bounds[XMIN]) {
        // Forward second order.
        dfdx = -1.5 * lbm_data(i, j, ifield) +
               2.0 * lbm_data(i + 1, j, ifield) -
               0.5 * lbm_data(i + 2, j, ifield);
      } else if (i >= bounds[XMAX]) {
        // Backward second order.
        dfdx = 1.5 * lbm_data(i, j, ifield) - 2.0 * lbm_data(i - 1, j, ifield) +
               0.5 * lbm_data(i - 2, j, ifield);
      } else {
        dfdx = 0.5 * (lbm_data(i + 1, j, ifield) - lbm_data(i - 1, j, ifield));
      }

      if (j <= bounds[YMIN]) {
        // Forward second order.
        dfdy = -1.5 * lbm_data(i, j, ifield) +
               2.0 * lbm_data(i, j + 1, ifield) -
               0.5 * lbm_data(i, j + 2, ifield);
      } else if (j >= bounds[YMAX]) {
        // Backward second order.
        dfdy = 1.5 * lbm_data(i, j, ifield) - 2.0 * lbm_data(i, j - 1, ifield) +
               0.5 * lbm_data(i, j - 2, ifield);
      } else {
        // Centered second order.
        dfdy = 0.5 * (lbm_data(i, j + 1, ifield) - lbm_data(i, j - 1, ifield));
      }
      dfdx /= dx;
      dfdy /= dx;
    } else // deal with bulk of domain
    {
      // compute directional derivatives of phi
      FState dfdir;
      for (int ipop = 0; ipop < npop; ++ipop) {
        const int i0 = this->lattice.E[ipop][IX];
        const int j0 = this->lattice.E[ipop][IY];
        dfdir[ipop] = (lbm_data(i + i0, j + j0, ifield) -
                       lbm_data(i - i0, j - j0, ifield)) /
                      (2 * dx);
        //~ dfdir[ipop] =
        //(lbm_data(i-2*i0,j-2*j0,ifield)-8*lbm_data(i-i0,j-j0,ifield) +
        //8*lbm_data(i+i0,j+j0,ifield) - lbm_data(i+2*i0,j+2*j0,ifield)) /
        //(12*dx);
      }
      // compute LBM gradients
      for (int ipop = 1; ipop < npop; ++ipop) {
        dfdx += w[ipop] * E[ipop][IX] * dfdir[ipop];
        dfdy += w[ipop] * E[ipop][IY] * dfdir[ipop];
      }
      dfdx *= e2;
      dfdy *= e2;
    }
    grad[IX] = dfdx;
    grad[IY] = dfdy;
  }

  // ================================================
  //  Compute gradient of ifield - 3d
  // ================================================
  KOKKOS_INLINE_FUNCTION
  void compute_gradient(RVect3 &grad, const IVect3 &IJK, int ifield,
                        int boundary) const {

    const real_t dx = params.dx;
    const real_t e2 = 3.0;

    int i = IJK[IX];
    int j = IJK[IY];
    int k = IJK[IZ];

    Kokkos::Array<int, 6> bounds(params.bounds[boundary]);
    const bool use_offcenter =
        ((i <= bounds[XMIN]) or (i >= bounds[XMAX]) or (j <= bounds[YMIN]) or
         (j >= bounds[YMAX]) or (k <= bounds[ZMIN]) or (k >= bounds[ZMAX]));
    real_t dfdx = 0.0, dfdy = 0.0, dfdz = 0.0;
    if (use_offcenter) {
      if (i <= bounds[XMIN]) {
        // Forward second order.
        dfdx = -1.5 * lbm_data(i, j, k, ifield) +
               2.0 * lbm_data(i + 1, j, k, ifield) -
               0.5 * lbm_data(i + 2, j, k, ifield);
      } else if (i >= bounds[XMAX]) {
        // Backward second order.
        dfdx = 1.5 * lbm_data(i, j, k, ifield) -
               2.0 * lbm_data(i - 1, j, k, ifield) +
               0.5 * lbm_data(i - 2, j, k, ifield);
      } else {
        dfdx = 0.5 *
               (lbm_data(i + 1, j, k, ifield) - lbm_data(i - 1, j, k, ifield));
      }

      if (j <= bounds[YMIN]) {
        // Forward second order.
        dfdy = -1.5 * lbm_data(i, j, k, ifield) +
               2.0 * lbm_data(i, j + 1, k, ifield) -
               0.5 * lbm_data(i, j + 2, k, ifield);
      } else if (j >= bounds[YMAX]) {
        // Backward second order.
        dfdy = 1.5 * lbm_data(i, j, k, ifield) -
               2.0 * lbm_data(i, j - 1, k, ifield) +
               0.5 * lbm_data(i, j - 2, k, ifield);
      } else {
        // Centered second order.
        dfdy = 0.5 *
               (lbm_data(i, j + 1, k, ifield) - lbm_data(i, j - 1, k, ifield));
      }

      if (k <= bounds[ZMIN]) {
        // Forward second order.
        dfdz = -1.5 * lbm_data(i, j, k, ifield) +
               2.0 * lbm_data(i, j, k + 1, ifield) -
               0.5 * lbm_data(i, j, k + 2, ifield);
      } else if (k >= bounds[ZMAX]) {
        // Backward second order.
        dfdz = 1.5 * lbm_data(i, j, k, ifield) -
               2.0 * lbm_data(i, j, k - 1, ifield) +
               0.5 * lbm_data(i, j, k - 2, ifield);
      } else {
        // Centered second order.
        dfdz = 0.5 *
               (lbm_data(i, j, k + 1, ifield) - lbm_data(i, j, k - 1, ifield));
      }

      dfdx /= dx;
      dfdy /= dx;
      dfdz /= dx;
    } else // deal with bulk of domain
    {

      // compute directional derivatives of phi
      FState dfdir;
      for (int ipop = 0; ipop < npop; ++ipop) {
        const int i0 = this->E[ipop][IX];
        const int j0 = this->E[ipop][IY];
        const int k0 = this->E[ipop][IZ];
        dfdir[ipop] = (lbm_data(i + i0, j + j0, k + k0, ifield) -
                       lbm_data(i - i0, j - j0, k - k0, ifield)) /
                      (2 * dx);
      }

      // compute LBM gradients
      for (int ipop = 1; ipop < npop; ++ipop) {
        dfdx += w[ipop] * E[ipop][IX] * dfdir[ipop];
        dfdy += w[ipop] * E[ipop][IY] * dfdir[ipop];
        dfdz += w[ipop] * E[ipop][IZ] * dfdir[ipop];
      }
      dfdx *= e2;
      dfdy *= e2;
      dfdz *= e2;

    } // end if offcenter else

    grad[IX] = dfdx;
    grad[IY] = dfdy;
    grad[IZ] = dfdz;

  } // compute_grad phi - 2d

  // ================================================
  //  Compute gradient of ifield - 2d
  // ================================================
  KOKKOS_INLINE_FUNCTION
  real_t compute_laplacian(const IVect2 &IJK, int ifield, int boundary) const {

    const real_t dx = params.dx;
    const real_t e2 = 3.0;

    int i = IJK[IX];
    int j = IJK[IY];

    real_t laplacian = 0.0;
    real_t d2fdx = 0.0, d2fdy = 0.0;

    Kokkos::Array<int, 6> bounds(params.bounds[boundary]);
    //~ int use_CDF2=0;
    //~ const bool use_offcenter = (i <= bounds[XMIN]+use_CDF2 or i >=
    //bounds[XMAX]-use_CDF2 or j <= bounds[YMIN]+use_CDF2 or j >=
    //bounds[YMAX]-use_CDF2);
    const bool use_offcenter = (i <= bounds[XMIN] or i >= bounds[XMAX] or
                                j <= bounds[YMIN] or j >= bounds[YMAX]);

    if (use_offcenter) {
      if (i <= bounds[XMIN]) {
        // Forward 2nd order.
        d2fdx =
            2.0 * lbm_data(i, j, ifield) - 5.0 * lbm_data(i + 1, j, ifield) +
            4.0 * lbm_data(i + 2, j, ifield) - 1.0 * lbm_data(i + 3, j, ifield);
      } else if (i >= bounds[XMAX]) {
        // Backward 2nd order.
        d2fdx =
            2.0 * lbm_data(i, j, ifield) - 5.0 * lbm_data(i - 1, j, ifield) +
            4.0 * lbm_data(i - 2, j, ifield) - 1.0 * lbm_data(i - 3, j, ifield);
      } else {
        // Centered 2nd order.
        d2fdx = lbm_data(i - 1, j, ifield) - 2.0 * lbm_data(i, j, ifield) +
                lbm_data(i + 1, j, ifield);
      }

      if (j <= bounds[YMIN]) {
        // Forward 2nd order.
        d2fdy =
            2.0 * lbm_data(i, j, ifield) - 5.0 * lbm_data(i, j + 1, ifield) +
            4.0 * lbm_data(i, j + 2, ifield) - 1.0 * lbm_data(i, j + 3, ifield);
      } else if (j >= bounds[YMAX]) {
        // Backward 2nd order.
        d2fdy =
            2.0 * lbm_data(i, j, ifield) - 5.0 * lbm_data(i, j - 1, ifield) +
            4.0 * lbm_data(i, j - 2, ifield) - 1.0 * lbm_data(i, j - 3, ifield);
      } else {
        // Centered 2nd order.
        d2fdy = lbm_data(i, j - 1, ifield) - 2.0 * lbm_data(i, j, ifield) +
                lbm_data(i, j + 1, ifield);
      }

      laplacian = (d2fdx + d2fdy) / SQR(dx);
    } else // deal with bulk of domain
    {

      for (int ipop = 0; ipop < npop; ++ipop) {
        const int i0 = E[ipop][IX];
        const int j0 = E[ipop][IY];
        const real_t d2f = lbm_data(i + i0, j + j0, ifield) -
                           2.0 * lbm_data(i, j, ifield) +
                           lbm_data(i - i0, j - j0, ifield);
        laplacian += e2 * w[ipop] * d2f / (dx * dx);
      }
    }
    return laplacian;
  }

  // ================================================
  //  Compute gradient of ifield - 3d
  // ================================================
  KOKKOS_INLINE_FUNCTION
  real_t compute_laplacian(const IVect3 &IJK, int ifield, int boundary) const {

    const real_t dx = params.dx;
    const real_t e2 = 3.0;

    int i = IJK[IX];
    int j = IJK[IY];
    int k = IJK[IZ];

    real_t laplacian = 0.0;
    real_t d2fdx = 0.0, d2fdy = 0.0, d2fdz = 0.0;

    Kokkos::Array<int, 6> bounds(params.bounds[boundary]);
    const bool use_offcenter =
        ((i <= bounds[XMIN]) or (i >= bounds[XMAX]) or (j <= bounds[YMIN]) or
         (j >= bounds[YMAX]) or (k <= bounds[ZMIN]) or (k >= bounds[ZMAX]));
    if (use_offcenter) {
      if (i <= bounds[XMIN]) {
        // Forward 2nd order.
        d2fdx = 2.0 * lbm_data(i, j, k, ifield) -
                5.0 * lbm_data(i + 1, j, k, ifield) +
                4.0 * lbm_data(i + 2, j, k, ifield) -
                1.0 * lbm_data(i + 3, j, k, ifield);
      } else if (i >= bounds[XMAX]) {
        // Backward 2nd order.
        d2fdx = 2.0 * lbm_data(i, j, k, ifield) -
                5.0 * lbm_data(i - 1, j, k, ifield) +
                4.0 * lbm_data(i - 2, j, k, ifield) -
                1.0 * lbm_data(i - 3, j, k, ifield);
      } else {
        // Centered 2nd order.
        d2fdx = lbm_data(i - 1, j, k, ifield) -
                2.0 * lbm_data(i, j, k, ifield) + lbm_data(i + 1, j, k, ifield);
      }
      if (j <= bounds[YMIN]) {
        // Forward 2nd order.
        d2fdy = 2.0 * lbm_data(i, j, k, ifield) -
                5.0 * lbm_data(i, j + 1, k, ifield) +
                4.0 * lbm_data(i, j + 2, k, ifield) -
                1.0 * lbm_data(i, j + 3, k, ifield);
      } else if (j >= bounds[YMAX]) {
        // Backward 2nd order.
        d2fdy = 2.0 * lbm_data(i, j, k, ifield) -
                5.0 * lbm_data(i, j - 1, k, ifield) +
                4.0 * lbm_data(i, j - 2, k, ifield) -
                1.0 * lbm_data(i, j - 3, k, ifield);
      } else {
        // Centered 2nd order.
        d2fdy = lbm_data(i, j - 1, k, ifield) -
                2.0 * lbm_data(i, j, k, ifield) + lbm_data(i, j + 1, k, ifield);
      }
      if (k <= bounds[ZMIN]) {
        // Forward 2nd order.
        d2fdz = 2.0 * lbm_data(i, j, k, ifield) -
                5.0 * lbm_data(i, j, k + 1, ifield) +
                4.0 * lbm_data(i, j, k + 2, ifield) -
                1.0 * lbm_data(i, j, k + 3, ifield);
      } else if (k >= bounds[ZMAX]) {
        // Backward 2nd order.
        d2fdz = 2.0 * lbm_data(i, j, k, ifield) -
                5.0 * lbm_data(i, j, k - 1, ifield) +
                4.0 * lbm_data(i, j, k - 2, ifield) -
                1.0 * lbm_data(i, j, k - 3, ifield);
      } else {
        // Centered 2nd order.
        d2fdz = lbm_data(i, j, k - 1, ifield) -
                2.0 * lbm_data(i, j, k, ifield) + lbm_data(i, j, k + 1, ifield);
      }

      laplacian = (d2fdx + d2fdy + d2fdz) / SQR(dx);
    } else // deal with bulk of domain
    {

      for (int ipop = 0; ipop < npop; ++ipop) {
        const int i0 = E[ipop][IX];
        const int j0 = E[ipop][IY];
        const int k0 = E[ipop][IZ];
        const real_t d2f = lbm_data(i + i0, j + j0, k + k0, ifield) -
                           2.0 * lbm_data(i, j, k, ifield) +
                           lbm_data(i - i0, j - j0, k - k0, ifield);
        laplacian += e2 * w[ipop] * d2f / (dx * dx);
      }

    } // end if offcenter else

    return laplacian;

  } // compute_grad phi - 2d

  //================================================
  // Boundary conditions
  //================================================
  // check if a direction should be reflected by the wall
  KOKKOS_INLINE_FUNCTION
  bool reflect(int faceId, int ipop) const {
    return ((faceId == FACE_XMIN and E[ipop][IX] > 0) or
            (faceId == FACE_XMAX and E[ipop][IX] < 0) or
            (faceId == FACE_YMIN and E[ipop][IY] > 0) or
            (faceId == FACE_YMAX and E[ipop][IY] < 0) or
            (faceId == FACE_ZMIN and E[ipop][IZ] > 0) or
            (faceId == FACE_ZMAX and E[ipop][IZ] < 0));
  }

  // anti bounce back
  template <typename EquationTagTemp>
  KOKKOS_INLINE_FUNCTION void
  compute_boundary_antibounceback(EquationTagTemp tag, int faceId,
                                  const IVect<dim> &IJK, int ipop,
                                  real_t value) const {
    IVect<dim> IJKs;
    bool can_str = stream_alldir(IJK, IJKs, ipop);
    if (can_str and reflect(faceId, ipop)) {
      real_t val = -get_f_val(tag, IJK, Ebar[ipop]) + 2.0 * w[ipop] * value;
      set_f_val(tag, IJKs, ipop, val);
    }
  }

  // bounce back
  template <typename EquationTagTemp>
  KOKKOS_INLINE_FUNCTION void
  compute_boundary_bounceback(EquationTagTemp tag, int faceId,
                              const IVect<dim> &IJK, int ipop,
                              real_t value) const {
    IVect<dim> IJKs;
    bool can_str = stream_alldir(IJK, IJKs, ipop);
    //~ if (can_str and reflect(faceId, ipop))
    if (can_str and reflect(faceId, ipop)) {
      real_t val = get_f_val(tag, IJK, Ebar[ipop]) - 2.0 * w[ipop] * value;

      set_f_val(tag, IJKs, ipop, val);
    }
  }

  // utils for connected components labelling
  KOKKOS_INLINE_FUNCTION
  int IJKToCCindex(const IVect<dim> &IJK) const {

    int isize = params.isize;
    int jsize = params.jsize;
    int ksize = params.ksize;
    // MPI domain coordinates
#ifdef USE_MPI
        const int i_mpi = params.myMpiPos[IX];
        const int j_mpi = params.myMpiPos[IY];
        const int k_mpi = params.myMpiPos[IZ];
        const int mx = params.mx;
        const int my = params.my;
#else
        const int i_mpi = 0;
        const int j_mpi = 0;
        const int k_mpi = 0;
        const int mx = 1;
        const int my = 1;
#endif
        real_t i = dim == 2
                     ? 1 + (IJK[0] + i_mpi * isize) +
                         (isize * mx) * (IJK[1] + j_mpi * jsize)
                     : 1 + (IJK[0] + i_mpi * isize) +
                         (isize * mx) * (IJK[1] + j_mpi * jsize) +
                         (isize * mx) * (jsize * my) * (IJK[2] + k_mpi * ksize);
        return i;
    }

    KOKKOS_INLINE_FUNCTION
    IVect2 CCindexToIVect2(const int idx) const
    {
        const int isize = params.isize;
        const int jsize = params.jsize;
        // MPI domain coordinates
#ifdef USE_MPI
        const int i_mpi = params.myMpiPos[IX];
        const int j_mpi = params.myMpiPos[IY];
        const int mx = params.mx;
#else
        const int i_mpi = 0;
        const int j_mpi = 0;
        const int mx = 1;
#endif

        const int sourcej = ((idx - 1) / (isize * mx)) - j_mpi * jsize;
        const int sourcei =
          int(idx) - 1 - i_mpi * isize - isize * mx * (sourcej + j_mpi * jsize);
        IVect2 source_pos({ sourcei, sourcej });
        return source_pos;
    }

    KOKKOS_INLINE_FUNCTION
    IVect3 CCindexToIVect3(const int idx) const
    {
        const int isize = params.isize;
        const int jsize = params.jsize;
        const int ksize = params.ksize;
        // MPI domain coordinates
#ifdef USE_MPI
        const int i_mpi = params.myMpiPos[IX];
        const int j_mpi = params.myMpiPos[IY];
        const int k_mpi = params.myMpiPos[IZ];
        const int mx = params.mx;
        const int my = params.my;
#else
        const int i_mpi = 0;
        const int j_mpi = 0;
        const int k_mpi = 0;
        const int mx = 1;
        const int my = 1;
#endif

        const int sourcek = ((idx)-1) / (isize * mx * jsize * my) - k_mpi * ksize;
        const int sourcej =
          (int(idx) - 1 - isize * mx * jsize * my * (sourcek + k_mpi * ksize)) /
            (isize * mx) -
          j_mpi * jsize;
        const int sourcei = idx - 1 -
                            isize * mx * jsize * my * (sourcek + k_mpi * ksize) -
                            (isize * mx) * (sourcej + j_mpi * jsize) - i_mpi * isize;

        IVect3 source_pos({ sourcei, sourcej, sourcek });
        return source_pos;
    }

}; // end class LBMSchemeBase

#endif // LBM_SCHEME_BASE_H_
