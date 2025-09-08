#ifndef CUSTOM_FUNCTOR_H_
#define CUSTOM_FUNCTOR_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>

#endif // __CUDA_ARCH__

#include "LBM_Base_Functor.h"

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Apply a custom operation on the entire domain (mostly for operations on macro vars)
 *
 * The custom operation is applied through a functor whose type is passed as template argument : FunctorCustomOp
 * These functor should be defined as part of the LBMscheme class, and must respect two condition:
 * 1. They have an attribute of type LBMscheme, and the constructor takes the scheme as argument to init this attribute.
 * 2. They have the method "void operator()(IVect<dim> IJK)", which executes the operation.
 *      This method can use anything available to scheme, such as get_LBMval, LBMparams and Model attributes, etc...
 *
 */
template <int dim, int npop, typename LBMSchemeSpec, typename FunctorCustomOp>
class CustomFunctor {

public:
    using LBMArray = typename LBMBaseFunctor<dim, npop>::LBMArray;
    using FArray = typename LBMBaseFunctor<dim, npop>::FArray;
    using FArrayConst = typename LBMBaseFunctor<dim, npop>::FArrayConst;
    using LBM_Weights = typename LBMBaseFunctor<dim, npop>::LBM_Weights;
    using LBM_speeds = typename LBMBaseFunctor<dim, npop>::LBM_speeds;
    using FState = typename Kokkos::Array<real_t, npop>;

    //~ typedef  void (LBMSchemeSpec::*SchemeCustomFunc)(const IVect<dim>&) const;

    //! attributes
    int isize, jsize, ksize, gw;
    LBMSchemeSpec scheme;
    //~ SchemeCustomFunc customFuncPtr;
    FunctorCustomOp functor_custom;




    CustomFunctor(const LBMSchemeSpec& scheme, int isize, int jsize, int ksize, int ghostWidth)
        : isize(isize)
        , jsize(jsize)
        , ksize(ksize)
        , gw(ghostWidth)
        , scheme(scheme)
        , functor_custom(scheme)
         {};

    // static method which does it all: create and execute functor

    inline static void apply(const LBMParams& params, const LBMSchemeSpec& scheme)
    {

        const int nbCells = dim == 2 ? params.isize * params.jsize : params.isize * params.jsize * params.ksize;

        CustomFunctor functor(scheme, params.isize, params.jsize, params.ksize, params.ghostWidth);
        Kokkos::RangePolicy policy(0, nbCells);
        Kokkos::parallel_for(policy, functor);
    }

    // ================================================================================================
    //
    // 2D version.
    //
    // ================================================================================================

    template <int dim_ = dim>
    KOKKOS_INLINE_FUNCTION void operator()(const typename std::enable_if<dim_ == 2, int>::type& index) const
    {
        IVect<dim> IJK;
        index2coord(index, IJK, isize, jsize);
        //~ (scheme.*customFuncPtr)(IJK);
        functor_custom(IJK);
    }



    // ================================================================================================
    //
    // 3D version.
    //
    // ================================================================================================

    //! functor for 3d - update macro
    template <int dim_ = dim>
    KOKKOS_INLINE_FUNCTION void operator()(const typename std::enable_if<dim_ == 3, int>::type index) const
    {
        IVect<dim> IJK;
        index2coord(index, IJK, isize, jsize, ksize);
        //~ (scheme.*customFuncPtr)(IJK);
        functor_custom(IJK);
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
