#ifndef MIN_FUNCTOR_H_
#define MIN_FUNCTOR_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>

#endif // __CUDA_ARCH__

#include "FieldManager.h"
#include "LBM_Base_Functor.h"

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Update distribution function for phase field problem.
 *
 * Two types of operation are allowed:
 * - perform a full Collide and Stream operation (LBM time update t -> t+dt)
 * - just compute equilibrium functions (in f0)
 *
 *
 */
template <int dim, int npop, typename LBMSchemeSpec, typename EquationTag>
class MinReducer {

public:
    using LBMArray = typename LBMBaseFunctor<dim, npop>::LBMArray;
    using FArray = typename LBMBaseFunctor<dim, npop>::FArray;
    using FArrayConst = typename LBMBaseFunctor<dim, npop>::FArrayConst;
    using LBM_Weights = typename LBMBaseFunctor<dim, npop>::LBM_Weights;
    using LBM_speeds = typename LBMBaseFunctor<dim, npop>::LBM_speeds;
    using FState = typename Kokkos::Array<real_t, npop>;

    using value_type = real_t;

    //! attributes
    int isize, jsize, ksize, gw;
    int ifield;
    LBMSchemeSpec scheme;
    EquationTag tag;

    /**
	 * Update distribution function for the phase_field problem.
	 */

    MinReducer(const LBMSchemeSpec& scheme, int isize, int jsize, int ksize, int ghostWidth, int ifield)
        : isize(isize)
        , jsize(jsize)
        , ksize(ksize)
        , gw(ghostWidth)
        , ifield(ifield)
        , scheme(scheme) {};

    // static method which does it all: create and execute functor
    inline static void apply(const LBMParams& params, const LBMSchemeSpec& scheme, int ifield, real_t& minR)
    {

        const int nbCells = dim == 2 ? params.isize * params.jsize : params.isize * params.jsize * params.ksize;

        MinReducer functor(scheme, params.isize, params.jsize, params.ksize, params.ghostWidth, ifield);

        real_t min;

        Kokkos::RangePolicy<int> policy(0, nbCells);
        Kokkos::parallel_reduce("MinReduce", policy, functor, Kokkos::Min<real_t>(min));

        minR = min;
    }

    // ================================================================================================
    //
    // 2D version.
    //
    // ================================================================================================

    template <int dim_ = dim>
    KOKKOS_INLINE_FUNCTION void operator()(const typename std::enable_if<dim_ == 2, int>::type& index, typename std::enable_if<dim_ == 2, real_t>::type& lmin) const
    {
        IVect<dim> IJK;
        index2coord(index, IJK, isize, jsize);
        real_t val = scheme.get_lbm_val(IJK, ifield);
        if ((val < lmin) && is_in_bounds(IJK))
            lmin = val;
    }

    //~ KOKKOS_INLINE_FUNCTION
    //~ void operator()(const int& index, real_t& lmin) const
    //~ {
    //~ IVect2 IJK;
    //~ index2coord(index,IJK,isize,jsize);
    //~ real_t val = scheme.get_lbm_val(IJK, ifield);
    //~ if( val < lmin ) lmin = val;
    //~ }

    // ================================================================================================
    //
    // 3D version.
    //
    // ================================================================================================

    template <int dim_ = dim>
    KOKKOS_INLINE_FUNCTION void operator()(const typename std::enable_if<dim_ == 3, int>::type& index, typename std::enable_if<dim_ == 3, real_t>::type& lmin) const
    {
        IVect<dim> IJK;
        index2coord(index, IJK, isize, jsize, ksize);
        real_t val = scheme.get_lbm_val(IJK, ifield);
        if ((val < lmin) && is_in_bounds(IJK))
            lmin = val;
    }

    // ================================================================================================
    //
    // reducer special functions
    //
    // ================================================================================================
    KOKKOS_INLINE_FUNCTION
    void join(volatile real_t& dst, const volatile real_t& src) const
    { // min operation
        if (dst > src) {
            dst = src;
        }
    }

    // Tell each thread how to initialize its reduction result.
    KOKKOS_INLINE_FUNCTION
    void init(real_t& dst) const
    { // The identity under max is -Inf.
        //~ dst = reduction_identity<value_type>::max();
        dst = 0.0;
    }
    // ================================================================================================
    //
    // additionnal functions
    //
    // ================================================================================================
    KOKKOS_INLINE_FUNCTION
    void index2coord(int index, IVect2& IJK, int Nx, int Ny) const
    {
#ifdef KOKKOS_ENABLE_CUDA
        IJK[1] = index / Nx;
        IJK[0] = index - IJK[1] * Nx;
#else
        IJK[0] = index / Ny;
        IJK[1] = index - IJK[0] * Ny;
#endif
    }

    KOKKOS_INLINE_FUNCTION
    void index2coord(int index, IVect3& IJK, int Nx, int Ny, int Nz) const
    {
        UNUSED(Nx);
        UNUSED(Nz);
#ifdef KOKKOS_ENABLE_CUDA
        int NxNy = Nx * Ny;
        IJK[2] = index / NxNy;
        IJK[1] = (index - IJK[2] * NxNy) / Nx;
        IJK[0] = index - IJK[1] * Nx - IJK[2] * NxNy;
#else
        int NyNz = Ny * Nz;
        IJK[0] = index / NyNz;
        IJK[1] = (index - IJK[0] * NyNz) / Nz;
        IJK[2] = index - IJK[1] * Nz - IJK[0] * NyNz;
#endif
    }

    // return versions
    KOKKOS_INLINE_FUNCTION
    IVect2 index2coord(int index, int Nx, int Ny) const
    {
        IVect2 IJK;
#ifdef KOKKOS_ENABLE_CUDA
        IJK[1] = index / Nx;
        IJK[0] = index - IJK[1] * Nx;
#else
        IJK[0] = index / Ny;
        IJK[1] = index - IJK[0] * Ny;
#endif
        return IJK;
    }

    KOKKOS_INLINE_FUNCTION
    IVect3 index2coord(int index, int Nx, int Ny, int Nz) const
    {
        IVect3 IJK;
        UNUSED(Nx);
        UNUSED(Nz);
#ifdef KOKKOS_ENABLE_CUDA
        int NxNy = Nx * Ny;
        IJK[2] = index / NxNy;
        IJK[1] = (index - IJK[2] * NxNy) / Nx;
        IJK[0] = index - IJK[1] * Nx - IJK[2] * NxNy;
#else
        int NyNz = Ny * Nz;
        IJK[0] = index / NyNz;
        IJK[1] = (index - IJK[0] * NyNz) / Nz;
        IJK[2] = index - IJK[1] * Nz - IJK[0] * NyNz;
#endif
        return IJK;
    }
    // ================================================
    //  check if we are inside the domain to update (because ghost cells should not stream inside the domain unless the boundary is periodic)
    // ================================================
    // is_in_bounds 2d
    KOKKOS_INLINE_FUNCTION
    bool is_in_bounds(IVect2& IJK) const
    {
        return (IJK[IX] >= scheme.get_bound(tag, FACE_XMIN)
            and IJK[IX] <= scheme.get_bound(tag, FACE_XMAX)
            and IJK[IY] >= scheme.get_bound(tag, FACE_YMIN)
            and IJK[IY] <= scheme.get_bound(tag, FACE_YMAX));
    }

    // is_in_bounds 3d
    KOKKOS_INLINE_FUNCTION
    bool is_in_bounds(IVect3& IJK) const
    {
        return (IJK[IX] >= scheme.get_bound(tag, FACE_XMIN)
            and IJK[IX] <= scheme.get_bound(tag, FACE_XMAX)
            and IJK[IY] >= scheme.get_bound(tag, FACE_YMIN)
            and IJK[IY] <= scheme.get_bound(tag, FACE_YMAX)
            and IJK[IZ] >= scheme.get_bound(tag, FACE_ZMIN)
            and IJK[IZ] <= scheme.get_bound(tag, FACE_ZMAX));
    }
};

#endif // FUNCTOR_ECHEBARIA_MODEL_H_
