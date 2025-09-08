#ifndef LBM_LATTICE_H_
#define LBM_LATTICE_H_

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
template <int dim_t, int npop_t> class LBMLattice {

public:
  int dim;
  int npop;

  // Types
  using LBM_Weights = typename LBMBaseFunctor<dim_t, npop_t>::LBM_Weights;
  using LBM_speeds = typename LBMBaseFunctor<dim_t, npop_t>::LBM_speeds;
  using LBM_speeds_opposite = typename LBMBaseFunctor<dim_t, npop_t>::LBM_speeds_opposite;
  using Matrix = typename LBMBaseFunctor<dim_t, npop_t>::Matrix;

  // attributes
  //! LBM weights
  LBM_Weights w;

  //! array of lattice velocity directions.
  LBM_speeds E;
  LBM_speeds_opposite Ebar;

  //! M and its inverse
  Matrix M;
  Matrix Minv;

  LBMLattice()
      : dim(dim_t), npop(npop_t), w(LBMBaseFunctor<dim_t, npop_t>::w),
        E(LBMBaseFunctor<dim_t, npop_t>::E),
        Ebar(LBMBaseFunctor<dim_t, npop_t>::Ebar),
        M(LBMBaseFunctor<dim_t, npop_t>::M),
        Minv(LBMBaseFunctor<dim_t, npop_t>::Minv){

        };

}; // class LBMLattice

#endif // LBM_LATTICE_H_
