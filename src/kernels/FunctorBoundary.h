#ifndef BOUNDARY_FUNCTOR_H_
#define BOUNDARY_FUNCTOR_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>

#endif // __CUDA_ARCH__

#include "LBMParams.h"
#include "LBM_Base_Functor.h"

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Apply boundary conditions to distribution functions by calling the boundary scheme
 *
 *
 *
 */
template <int dim, int npop, typename LBMSchemeSpec>
class BoundaryFunctor {

public:
    using LBMArray = typename LBMBaseFunctor<dim, npop>::LBMArray;
    using FArray = typename LBMBaseFunctor<dim, npop>::FArray;
    using FArrayConst = typename LBMBaseFunctor<dim, npop>::FArrayConst;
    using LBM_Weights = typename LBMBaseFunctor<dim, npop>::LBM_Weights;
    using LBM_speeds = typename LBMBaseFunctor<dim, npop>::LBM_speeds;
    using FState = typename Kokkos::Array<real_t, npop>;

    //==========================
    // attributes
    int isize, jsize, ksize;
    int nx, ny, nz;
    int ghostWidth;
    int faceId, faceSize;
    LBMSchemeSpec scheme;

    BoundaryFunctor(const LBMSchemeSpec& scheme, const LBMParams& params, FaceIdType faceId, int faceSize)
        : isize(params.isize)
        , jsize(params.jsize)
        , ksize(params.ksize)
        , nx(params.nx)
        , ny(params.ny)
        , nz(params.nz)
        , ghostWidth(params.ghostWidth)
        , faceId(faceId)
        , faceSize(faceSize)
        , scheme(scheme) {};
    // static method calls every boundary
    inline static void make_all_boundaries(const LBMParams& params, const LBMSchemeSpec& scheme)
    {

        bool proc_is_xmin_boundary = true;
        bool proc_is_xmax_boundary = true;
        bool proc_is_ymin_boundary = true;
        bool proc_is_ymax_boundary = true;
        bool proc_is_zmin_boundary = true;
        bool proc_is_zmax_boundary = true;

#ifdef USE_MPI
        //~ printf( "checking mpi neighbors\n");
        proc_is_xmin_boundary = (params.myMpiPos[DIR_X] == 0);
        proc_is_xmax_boundary = (params.myMpiPos[DIR_X] == params.mx - 1);
        proc_is_ymin_boundary = (params.myMpiPos[DIR_Y] == 0);
        proc_is_ymax_boundary = (params.myMpiPos[DIR_Y] == params.my - 1);
        proc_is_zmin_boundary = (params.myMpiPos[DIR_Z] == 0);
        proc_is_zmax_boundary = (params.myMpiPos[DIR_Z] == params.mz - 1);
#endif // USE_MPI

        // send boundary functor for each face if it faces an external non periodic boundary
        if (proc_is_xmin_boundary) {
            apply(params, scheme, FACE_XMIN);
        } //else {printf("rank %d,skipping xmin boundary\n", params.myRank);}
        if (proc_is_xmax_boundary) {
            apply(params, scheme, FACE_XMAX);
        } //else {printf("rank %d,skipping xmax boundary\n", params.myRank);}
        if (proc_is_ymin_boundary) {
            apply(params, scheme, FACE_YMIN);
        } //else {printf("rank %d,skipping ymin boundary\n", params.myRank);}
        if (proc_is_ymax_boundary) {
            apply(params, scheme, FACE_YMAX);
        } //else {printf("rank %d,skipping ymax boundary\n", params.myRank);}

        if (dim == 3) {
            if (proc_is_zmin_boundary) {
                apply(params, scheme, FACE_ZMIN);
            }
            if (proc_is_zmax_boundary) {
                apply(params, scheme, FACE_ZMAX);
            }
        }
    }

    // static method which does it all: create and execute functor
    inline static void apply(const LBMParams& params, const LBMSchemeSpec& scheme, FaceIdType faceId)
    {
        int faceSize = 0;
        if (dim == 2) {
            if (faceId == FACE_XMAX or faceId == FACE_XMIN)
                faceSize = params.jsize;
            if (faceId == FACE_YMAX or faceId == FACE_YMIN)
                faceSize = params.isize;
        }
        if (dim == 3) {
            if (faceId == FACE_XMAX or faceId == FACE_XMIN)
                faceSize = params.jsize * params.ksize;
            if (faceId == FACE_YMAX or faceId == FACE_YMIN)
                faceSize = params.isize * params.ksize;
            if (faceId == FACE_ZMAX or faceId == FACE_ZMIN)
                faceSize = params.isize * params.jsize;
        }

        int nbIter = params.ghostWidth * faceSize;

        BoundaryFunctor functor(scheme, params, faceId, faceSize);

        Kokkos::parallel_for(nbIter, functor);
    }

    // ================================================
    //
    // 2D version.
    //
    // ================================================
    template <int dim_ = dim>
    KOKKOS_INLINE_FUNCTION void operator()(const typename std::enable_if<dim_ == 2, int>::type& index) const
    {
        IVect<dim> IJK;
        index2BorderCoord(index, IJK);

        scheme.make_boundary(IJK, faceId);

    } // operator() - 2d - boundary

    // ================================================
    //
    // 3D version.
    //
    // ================================================
    template <int dim_ = dim>
    KOKKOS_INLINE_FUNCTION void operator()(const typename std::enable_if<dim_ == 3, int>::type& index) const
    {
        IVect<dim> IJK;
        index2BorderCoord(index, IJK);

        scheme.make_boundary(IJK, faceId);
    }

    // ================================================
    //
    // additionnal functions
    //
    // ================================================
    // get 2d border coordinates
    KOKKOS_INLINE_FUNCTION
    void index2BorderCoord(int index, IVect2& IJK) const
    {
        if (faceId == FACE_XMIN) {

            IJK[IY] = index / ghostWidth;
            IJK[IX] = ghostWidth - 1 - (index % ghostWidth);
            //~ printf("face xmin boundary at (%d,%d)\n", IJK[IX], IJK[IY]);
        } else if (faceId == FACE_XMAX) {

            IJK[IY] = index / ghostWidth;
            IJK[IX] = index % ghostWidth;
            IJK[IX] += (nx + ghostWidth);
            //~ printf("face xmax boundary at (%d,%d)\n", IJK[IX], IJK[IY]);
        } else if (faceId == FACE_YMIN) {

            IJK[IX] = index / ghostWidth;
            IJK[IY] = ghostWidth - 1 - (index % ghostWidth);
        } else if (faceId == FACE_YMAX) {

            IJK[IX] = index / ghostWidth;
            IJK[IY] = index % ghostWidth;
            IJK[IY] += (ny + ghostWidth);
        }

        else {
            IJK[IX] = -1;
            IJK[IY] = -1;
        }
    }

    // get 3d border coordinates
    KOKKOS_INLINE_FUNCTION
    void index2BorderCoord(int index, IVect3& IJK) const
    {
        int i = 0, j = 0, k = 0;
        if (faceId == FACE_XMIN) {

            k = index / (ghostWidth * jsize);
            j = (index - k * ghostWidth * jsize) / ghostWidth;
            i = index - j * ghostWidth - k * ghostWidth * jsize;
        } else if (faceId == FACE_XMAX) {

            k = index / (ghostWidth * jsize);
            j = (index - k * ghostWidth * jsize) / ghostWidth;
            i = index - j * ghostWidth - k * ghostWidth * jsize;
            i += (nx + ghostWidth);

        } else if (faceId == FACE_YMIN) {

            k = index / (isize * ghostWidth);
            j = (index - k * isize * ghostWidth) / isize;
            i = index - j * isize - k * isize * ghostWidth;
        } else if (faceId == FACE_YMAX) {

            k = index / (isize * ghostWidth);
            j = (index - k * isize * ghostWidth) / isize;
            i = index - j * isize - k * isize * ghostWidth;
            j += (ny + ghostWidth);
        } else if (faceId == FACE_ZMIN) {
            k = index / (isize * jsize);
            j = (index - k * isize * jsize) / isize;
            i = index - j * isize - k * isize * jsize;
        } else if (faceId == FACE_ZMAX) {

            k = index / (isize * jsize);
            j = (index - k * isize * jsize) / isize;
            i = index - j * isize - k * isize * jsize;
            k += (nz + ghostWidth);
        }
        
//~ #ifdef KOKKOS_ENABLE_CUDA
        //~ int NxNy = Nx * Ny;
        //~ IJK[2] = index / NxNy;
        //~ IJK[1] = (index - IJK[2] * NxNy) / Nx;
        //~ IJK[0] = index - IJK[1] * Nx - IJK[2] * NxNy;
//~ #else
        //~ int NyNz = Ny * Nz;
        //~ IJK[0] = index / NyNz;
        //~ IJK[1] = (index - IJK[0] * NyNz) / Nz;
        //~ IJK[2] = index - IJK[1] * Nz - IJK[0] * NyNz;
//~ #endif

        IJK[IX] = i;
        IJK[IY] = j;
        IJK[IZ] = k;
    }

    // is_in_bounds 2d
    KOKKOS_INLINE_FUNCTION
    bool is_in_bounds(IVect2 IJK) const
    {
        return (IJK[IX] >= 0 and IJK[IX] <= isize - 1 and IJK[IY] >= 0 and IJK[IY] <= jsize - 1);
    }

    // is_in_bounds 3d
    KOKKOS_INLINE_FUNCTION
    bool is_in_bounds(const IVect3& IJK) const
    {
        return (IJK[IX] >= 0 and IJK[IX] <= isize - 1 and IJK[IY] >= 0 and IJK[IY] <= jsize - 1 and IJK[IZ] >= 0 and IJK[IZ] <= ksize - 1);
    }
};

#endif
