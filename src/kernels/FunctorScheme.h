#ifndef SCHEME_FUNCTOR_H_
#define SCHEME_FUNCTOR_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>

#endif // __CUDA_ARCH__

#include "LBM_Base_Functor.h"
#include "LBM_Lattice.h"

struct TagUpdate {
};
struct TagInit {
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
template <int dim, int npop, typename LBMSchemeSpec, typename EquationTag, typename Collider>
class SchemeFunctor {

public:
    using LBMArray = typename LBMBaseFunctor<dim, npop>::LBMArray;
    using FArray = typename LBMBaseFunctor<dim, npop>::FArray;
    using FArrayConst = typename LBMBaseFunctor<dim, npop>::FArrayConst;
    using LBM_Weights = typename LBMBaseFunctor<dim, npop>::LBM_Weights;
    using LBM_speeds = typename LBMBaseFunctor<dim, npop>::LBM_speeds;
    using FState = typename Kokkos::Array<real_t, npop>;

    //! attributes
    int isize, jsize, ksize, gw;
    LBMLattice<dim, npop> lattice;
    LBMSchemeSpec scheme;
    EquationTag tag;

    /**
     * Update distribution function for the phase_field problem.
     */

    SchemeFunctor(const LBMSchemeSpec& scheme, int isize, int jsize, int ksize, int ghostWidth)
        : isize(isize)
        , jsize(jsize)
        , ksize(ksize)
        , gw(ghostWidth)
        , lattice(LBMLattice<dim, npop>())
        , scheme(scheme) {};

    // static method which does it all: create and execute functor
    template <typename TagAction>
    inline static void apply(const LBMParams& params, const LBMSchemeSpec& scheme)
    {
        const int isize = params.isize;
        const int jsize = params.jsize;
        const int ksize = params.ksize;
        const int nbCells = dim == 2 ? isize * jsize : isize * jsize * ksize;

        SchemeFunctor functor(scheme, isize, jsize, ksize, params.ghostWidth);

        Kokkos::RangePolicy<TagAction> policy(0, nbCells);

        Kokkos::parallel_for(policy, functor);
    }

    // ================================================================================================
    //
    // 2D version.
    //
    // ================================================================================================
    //! functor for 2d - full update phi
    template <int dim_ = dim>
    KOKKOS_INLINE_FUNCTION void operator()(const TagUpdate, const typename std::enable_if<dim_ == 2, int>::type& index) const
    {
        //~ IVect<dim> IJK;
        //~ index2coord(index,IJK,isize,jsize);
        IVect2 IJK = index2coord(index, isize, jsize);

        if (is_in_bounds(IJK)) {
            //~ // compute feq and source terms
            Collider collider = Collider(lattice);
            scheme.setup_collider(tag, IJK, collider);
            // collide and stream
            scheme.stream(IJK, collider);
        }
        //~ else printf("not in bounds (scheme)\n");
    }

    template <int dim_ = dim>
    KOKKOS_INLINE_FUNCTION void operator()(const TagInit, const typename std::enable_if<dim_ == 2, int>::type& index) const
    {
        IVect<dim> IJK;
        index2coord(index, IJK, isize, jsize);

        Collider collider = Collider(lattice);
        scheme.setup_collider(tag, IJK, collider);
        scheme.set_f_as_feq(tag, IJK, collider);
    }

    // ================================================================================================
    //
    // 3D version.
    //
    // ================================================================================================
    //! functor for 3d
    template <int dim_ = dim>
    KOKKOS_INLINE_FUNCTION void operator()(const TagUpdate, const typename std::enable_if<dim_ == 3, int>::type& index) const
    {
        IVect<dim> IJK;
        index2coord(index, IJK, isize, jsize, ksize);

        if (is_in_bounds(IJK)) {
            Collider collider = Collider(lattice);
            scheme.setup_collider(tag, IJK, collider);
            scheme.stream(IJK, collider);
        }

    } // operator () - 3d - full update

    //! functor for 3d - compute feq phi
    template <int dim_ = dim>
    KOKKOS_INLINE_FUNCTION void operator()(const TagInit, const typename std::enable_if<dim_ == 3, int>::type& index) const
    {
        IVect<dim> IJK;
        index2coord(index, IJK, isize, jsize, ksize);

        Collider collider = Collider(lattice);
        scheme.setup_collider(tag, IJK, collider);
        scheme.set_f_as_feq(tag, IJK, collider);

    } // operator() - 3d - compute Feq

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
    bool is_in_bounds(const IVect2& IJK) const
    {
        return (IJK[IX] >= scheme.get_bound(tag, FACE_XMIN)
            and IJK[IX] <= scheme.get_bound(tag, FACE_XMAX)
            and IJK[IY] >= scheme.get_bound(tag, FACE_YMIN)
            and IJK[IY] <= scheme.get_bound(tag, FACE_YMAX));
    }

    // is_in_bounds 3d
    KOKKOS_INLINE_FUNCTION
    bool is_in_bounds(const IVect3& IJK) const
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
