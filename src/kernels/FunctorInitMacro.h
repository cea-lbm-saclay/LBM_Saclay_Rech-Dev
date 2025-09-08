#ifndef INIT_MACRO_FUNCTOR_H_
#define INIT_MACRO_FUNCTOR_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>

#endif // __CUDA_ARCH__

#include "LBM_Base_Functor.h"

struct TagInitMacro {
};

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
template <int dim, int npop, typename LBMSchemeSpec>
class MacroInitFunctor {

public:
    using LBMArray = typename LBMBaseFunctor<dim, npop>::LBMArray;
    using FArray = typename LBMBaseFunctor<dim, npop>::FArray;
    using FArrayConst = typename LBMBaseFunctor<dim, npop>::FArrayConst;
    using LBM_Weights = typename LBMBaseFunctor<dim, npop>::LBM_Weights;
    using LBM_speeds = typename LBMBaseFunctor<dim, npop>::LBM_speeds;
    using FState = typename Kokkos::Array<real_t, npop>;

    //! attributes
    int isize, jsize, ksize, gw;
    LBMSchemeSpec scheme;

    RANDOM_POOL rand_pool;

    /**
     * Update distribution function for the phase_field problem.
     */

    MacroInitFunctor(LBMSchemeSpec& scheme, int isize, int jsize, int ksize, int ghostWidth)
        : isize(isize)
        , jsize(jsize)
        , ksize(ksize)
        , gw(ghostWidth)
        , scheme(scheme)
        , rand_pool(scheme.params.random_seed) {};

    // static method which does it all: create and execute functor
    template <typename TagAction>
    inline static void apply(LBMParams& params, LBMSchemeSpec& scheme)
    {

        const int nbCells = dim == 2 ? params.isize * params.jsize : params.isize * params.jsize * params.ksize;

        MacroInitFunctor functor(scheme, params.isize, params.jsize, params.ksize, params.ghostWidth);
        Kokkos::RangePolicy<TagAction> policy(0, nbCells);
        Kokkos::parallel_for(policy, functor);
    }

    // ================================================================================================
    //
    // 2D version.
    //
    // ================================================================================================
    template <int dim_ = dim>
    KOKKOS_INLINE_FUNCTION void operator()(const TagInitMacro, const typename std::enable_if<dim_ == 2, int>::type& index) const
    {

        IVect<dim> IJK;
        index2coord(index, IJK, isize, jsize);
        RANDOM_POOL::generator_type rand_gen = rand_pool.get_state();
        scheme.init_macro(IJK, rand_gen);
        real_t some_rand = rand_gen.drand();
        some_rand += 1;
        rand_pool.free_state(rand_gen);
    }

    // ================================================================================================
    //
    // 3D version.
    //
    // ================================================================================================
    //! functor for 3d - init macro
    template <int dim_ = dim>
    KOKKOS_INLINE_FUNCTION void operator()(const TagInitMacro, const typename std::enable_if<dim_ == 3, int>::type index) const
    {
        IVect<dim> IJK;
        index2coord(index, IJK, isize, jsize, ksize);
        RANDOM_POOL::generator_type rand_gen = rand_pool.get_state();
        scheme.init_macro(IJK, rand_gen);
        rand_pool.free_state(rand_gen);
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
};

#endif // FUNCTOR_ECHEBARIA_MODEL_H_
