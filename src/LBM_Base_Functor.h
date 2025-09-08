#ifndef LBM_BASE_FUNCTORS_H_
#define LBM_BASE_FUNCTORS_H_

#include <array>
#include <string>
#include <type_traits>

#include "LBMParams.h"
#include "kokkos_shared.h"
#include "real_type.h"

/**
 * Base class for any LBM compute kernels.
 *
 * Define alias type to useful data types.
 *
 * \tparam dim : 2 or 3
 * \tparam npop : number of populations
 * example: for D2Q9, npop=9
 */
template<int dim_t,
         int npop_t>
class LBMBaseFunctor
{

  public:
    //! make template parameter dimension a static member
    static const int dim = dim_t;

    //! make template parameter a static member
    static constexpr int npop = npop_t;

    //! define LBM_Weights type alias
    using LBM_Weights = Kokkos::Array<real_t, npop_t>;

    //! LBM weights
    static const LBM_Weights w;

    //! define LBM_speeds directions of velocity distributions
    using LBM_speeds = Kokkos::Array<Kokkos::Array<int, dim_t>, npop_t>;

    //! array of lattice velocity directions.
    static const LBM_speeds E;

    //! for a given population, return mathing population id corresponding
    //! the to opposite directions
    //! for example with D2Q5, 0 <-> 0, 1 <-> 3 and 2 <-> 4
    using LBM_speeds_opposite = Kokkos::Array<int, npop_t>;

    //! Matrix for projection to moment space needed for MRT Collider
    using Matrix = Kokkos::Array<Kokkos::Array<real_t, npop_t>, npop_t>;

    //! M and its inverse
    static const Matrix M, Minv;

    //! array of lattice velocity opposite directions.
    static const LBM_speeds_opposite Ebar;

    //! Decide at compile-time which data array to use
    using LBMArray = typename std::conditional<dim_t == 2, LBMArray2d, LBMArray3d>::type;
    using LBMArrayHost = typename std::conditional<dim_t == 2, LBMArray2dHost, LBMArray3dHost>::type;

    //! a const version of LBMArray
    using LBMArrayConst = typename std::conditional<dim_t == 2, LBMArrayConst2d, LBMArrayConst3d>::type;

    //! distribution functions array type is just an alias to LBMArray
    using FArray = typename std::conditional<dim_t == 2, FArray2d, FArray3d>::type;

    //! distribution functions const array type is just an alias to LBMArray
    using FArrayConst = typename std::conditional<dim_t == 2, FArrayConst2d, FArrayConst3d>::type;

    //! scalar array type definition
    using ArrayScalar = typename std::conditional<dim_t == 2, ArrayScalar2d, ArrayScalar3d>::type;

}; // class LBMBaseFunctor

#endif // LBM_BASE_FUNCTORS_H_
