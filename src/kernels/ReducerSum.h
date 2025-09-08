#ifndef SUM_FUNCTOR_H_
#define SUM_FUNCTOR_H_

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
 * Functor to compute the sum of a specific value over all cells in the domain
 * 
 * 
 * 
 * The value is computed through a functor whose type is passed as template argument : FunctorSummand
 * These functor should be defined as part of the LBMscheme class, and must respect two condition:
 * 1. They have an attribute of type LBMscheme, and the constructor takes the scheme as argument to init this attribute.
 * 2. They have the method "real_t operator()(IVect<dim> IJK)", which returns the value to be added to the global sum.
 * 		This method can use anything available to scheme, such as get_LBMval, LBMparams and Model attributes, etc...
 * 
 * 
 */

template <int dim, int npop, typename LBMSchemeSpec, typename FunctorSummand>
class ReducerSum {

public:
    using LBMArray = typename LBMBaseFunctor<dim, npop>::LBMArray;
    using FArray = typename LBMBaseFunctor<dim, npop>::FArray;
    using FArrayConst = typename LBMBaseFunctor<dim, npop>::FArrayConst;
    using LBM_Weights = typename LBMBaseFunctor<dim, npop>::LBM_Weights;
    using LBM_speeds = typename LBMBaseFunctor<dim, npop>::LBM_speeds;
    using FState = typename Kokkos::Array<real_t, npop>;


    
    //~ typedef  real_t (LBMSchemeSpec::*SchemeSumFunc)(const IVect<dim>&) const;
    
    //~ typedef decltype(LBMSchemeSpec::m_sumCA) SchemeSumFunc;

    //! attributes
    int isize, jsize, ksize, gw;
    
    LBMSchemeSpec scheme;
    //~ SchemeSumFunc func;
    FunctorSummand  funct_summand;

    /**
	 * Update distribution function for the phase_field problem.
	 */

    ReducerSum(const LBMSchemeSpec& scheme, int isize, int jsize, int ksize, int ghostWidth)
        : isize(isize)
        , jsize(jsize)
        , ksize(ksize)
        , gw(ghostWidth)
        , scheme(scheme)
        , funct_summand(scheme) 
        {};

    ~ReducerSum() {};

    // static method which does it all: create and execute functor
    inline static void apply(const LBMParams& params, const LBMSchemeSpec& scheme, real_t& sumR)
    {

        const int nbCells = dim == 2 ? params.isize * params.jsize : params.isize * params.jsize * params.ksize;

		//~ std::cout<<"red sum: create functor" << std::endl;
        ReducerSum functor(scheme, params.isize, params.jsize, params.ksize, params.ghostWidth);

        real_t sum;
	
        Kokkos::RangePolicy policy(0, nbCells);
        //~ std::cout<<"red sum: start parallel reduce" << std::endl;
        Kokkos::parallel_reduce(policy, functor, sum);

	
        sumR = sum;
        
        // reduction over all mpi proccesses:
        
		#ifdef USE_MPI
		
			const int data_type = params.data_type;

        
			real_t global_sum;
			
			params.communicator->allReduce(&sumR, &global_sum, 1, data_type,  hydroSimu::MpiComm::SUM);
			
			sumR=global_sum;
		#endif
		
		
        
        
        //~ std::cout<<"red sum: end" << std::endl;
    }

    // ================================================================================================
    //
    // 2D version.
    //
    // ================================================================================================

    template <int dim_ = dim>
    KOKKOS_INLINE_FUNCTION void operator()(const typename std::enable_if<dim_ == 2, int>::type& index, typename std::enable_if<dim_ == 2, real_t>::type& lsum) const
    {
        IVect<dim> IJK;
        index2coord(index, IJK, isize, jsize);


		if (scheme.isNotGhostCell(IJK))
            lsum = lsum+funct_summand(IJK);
            
        //~ lsum=0.0;
    }

    // ================================================================================================
    //
    // 3D version.
    //
    // ================================================================================================

    template <int dim_ = dim>
    KOKKOS_INLINE_FUNCTION void operator()(const typename std::enable_if<dim_ == 3, int>::type& index, typename std::enable_if<dim_ == 3, real_t>::type& lsum) const
    {
        IVect<dim> IJK;
        index2coord(index, IJK, isize, jsize, ksize);
       
        if (scheme.isNotGhostCell(IJK))
            lsum = lsum+funct_summand(IJK);
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
        //~ IJK[0] = index / Ny;
        //~ IJK[1] = index - IJK[0] * Ny;
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

    //~ // return versions
    //~ KOKKOS_INLINE_FUNCTION
    //~ IVect2 index2coord(int index, int Nx, int Ny) const
    //~ {
        //~ IVect2 IJK;
//~ #ifdef KOKKOS_ENABLE_CUDA
        //~ IJK[1] = index / Nx;
        //~ IJK[0] = index - IJK[1] * Nx;
//~ #else
        //~ IJK[0] = index / Ny;
        //~ IJK[1] = index - IJK[0] * Ny;
//~ #endif
        //~ return IJK;
    //~ }

    //~ KOKKOS_INLINE_FUNCTION
    //~ IVect3 index2coord(int index, int Nx, int Ny, int Nz) const
    //~ {
        //~ IVect3 IJK;
        //~ UNUSED(Nx);
        //~ UNUSED(Nz);
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
        //~ return IJK;
    //~ }
   
};

#endif
