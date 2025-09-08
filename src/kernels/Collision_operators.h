#ifndef COLLISION_OPERATORS_H_
#define COLLISION_OPERATORS_H_

// #include <limits> // for std::numeric_limits


#include "LBM_Base_Functor.h"
#include "LBM_Lattice.h"

struct EquationTag1 {};
struct EquationTag2 {};
struct EquationTag3 {};
struct EquationTag4 {};
struct EquationTag5 {};
struct EquationTag6 {};

template <int dim, int npop> struct BGKCollider {
public:
  using FState = typename Kokkos::Array<real_t, npop>;
  FState f;
  FState feq;
  FState S0;
  //~ FState S1;
  real_t tau;
  KOKKOS_INLINE_FUNCTION
  BGKCollider(const LBMLattice<dim, npop> &lattice){};

  KOKKOS_INLINE_FUNCTION
  real_t collide(int ipop) const {
    return (f[ipop] * (1.0 - 1.0 / tau) + feq[ipop] / tau + S0[ipop]);
  };
  KOKKOS_INLINE_FUNCTION
  real_t get_feq(int ipop) const { return feq[ipop]; };
};

template <int dim, int npop> struct BGKColliderTimeFactor {
public:
  using FState = typename Kokkos::Array<real_t, npop>;
  FState f;
  FState feq;
  FState S0;

  FState f_nonlocal;

  real_t tau;

  real_t factor;
  KOKKOS_INLINE_FUNCTION
  BGKColliderTimeFactor(const LBMLattice<dim, npop> &lattice){};

  KOKKOS_INLINE_FUNCTION
  real_t collide(int ipop) const {
    return ((f[ipop] - (1.0 - factor) * f_nonlocal[ipop] -
             (f[ipop] - feq[ipop]) / tau + S0[ipop]) /
            (FMAX(factor, 0.000000001)));
  }
  KOKKOS_INLINE_FUNCTION
  real_t get_feq(int ipop) const { return feq[ipop]; }
};

template <int dim, int npop> struct TRTCollider {
  using FState = typename Kokkos::Array<real_t, npop>;
  using LBM_speeds_opposite =
      typename LBMBaseFunctor<dim, npop>::LBM_speeds_opposite;
  LBM_speeds_opposite Ebar;

  FState f, feq;

  real_t tauS, tauA;

  FState S0;

  KOKKOS_INLINE_FUNCTION
  void setTau(real_t tau, int TRT_tauMethod) { tauA = tau; };

  KOKKOS_INLINE_FUNCTION
  TRTCollider(const LBMLattice<dim, npop> &lattice) { Ebar = lattice.Ebar; };
  //~ FState S1;
  KOKKOS_INLINE_FUNCTION
  real_t collide(int ipop) const {
    const int ipopb = Ebar[ipop];
    const real_t fi = f[ipop];
    const real_t fib = f[ipopb];
    const real_t feqi = feq[ipop];
    const real_t feqib = feq[ipopb];
    //~ real_t pS=0.5*((f[ipop]+f[ipopb])-(feq[ipop]+feq[ipopb]));
    //~ real_t pA=0.5*((f[ipop]-f[ipopb])-(feq[ipop]-feq[ipopb]));

    return (f[ipop] -
            0.5 * (((fi + fib) - (feqi + feqib)) / tauS +
                   ((fi - fib) - (feqi - feqib)) / tauA) +
            S0[ipop]);
  }
  // Function for NSAC_COMP
  KOKKOS_INLINE_FUNCTION
  real_t get_fTRT(int alpha) const {
    const int alphab = Ebar[alpha];
    const real_t fi = f[alpha];
    const real_t fib = f[alphab];
    const real_t feqi = feq[alpha];
    const real_t feqib = feq[alphab];

    return (fi - 0.5 * (((fi + fib) - (feqi + feqib)) / tauS +
                        ((fi - fib) - (feqi - feqib)) / tauA));
  }
  KOKKOS_INLINE_FUNCTION
  real_t get_feq(int ipop) const { return feq[ipop]; }
};

///////////////////////////////////////////////////////////////////////
//                              MRT
///////////////////////////////////////////////////////////////////////

template <int dim, int npop> struct MRTCollider {
  // FONCTIONS DE DISTRIBUTIONS
  using FState = typename Kokkos::Array<real_t, npop>;
  FState f;   // Disctribution courante
  FState feq; // Distribution à l'équilibre
  FState S0;  // Terme source

  // MATRICES MRT
  using Matrix = typename LBMBaseFunctor<dim, npop>::Matrix;
  const Matrix M, Minv;
  FState S;
  real_t tau;

  // Membres
  mutable bool already_calc;
  mutable FState Prod_invMSMf;

  KOKKOS_INLINE_FUNCTION
  MRTCollider(const LBMLattice<dim, npop> lattice)
      : M(lattice.M), Minv(lattice.Minv) {
    already_calc = false;
    for (int i = 0; i < npop; i++) {
      feq[i] = 0.0;
    }

    //~ print_Matrix(M);
    //~ print_Matrix(Minv);
  }

  KOKKOS_INLINE_FUNCTION
  void setMandMinv(const Matrix M_, const Matrix Minv_) {
    //~ M = M_; Minv = Minv_;
  }

  KOKKOS_INLINE_FUNCTION
  real_t collide(int ipop) const {

    if (!already_calc) {
      Calc_Lf();
      already_calc = true;
    }

    return f[ipop] - Prod_invMSMf[ipop] +
           S0[ipop]; //  - Prod_invMSMf[ipop] + S0[ipop];
  }

  KOKKOS_INLINE_FUNCTION
  real_t get_feq(int ipop) const { return feq[ipop]; }

  KOKKOS_INLINE_FUNCTION
  void Calc_Lf() const {
    FState f_feq;
    for (int ipop = 0; ipop < npop; ipop++) {
      f_feq[ipop] = f[ipop] - feq[ipop];
    }

    FState Prod_Mf = product_MatrixVector(M, f_feq);
    FState Prod_SMf = product_Vectors(S, Prod_Mf);
    Prod_invMSMf = product_MatrixVector(Minv, Prod_SMf);
    // Prod_invMSMf = f_feq;
  }

  KOKKOS_INLINE_FUNCTION
  FState product_MatrixVector(const Matrix &A, const FState &X) const {
    FState C;
    for (int i = 0; i < npop; i++) {
      real_t sum = 0.0;
      for (int j = 0; j < npop; j++) {
        sum = sum + A[i][j] * X[j];
      }
      C[i] = sum;
    }
    return C;
  }
  KOKKOS_INLINE_FUNCTION
  FState product_Vectors(const FState &U, const FState &V) const {
    FState C;
    for (int i = 0; i < npop; i++) {
      C[i] = 0;

      C[i] += U[i] * V[i];
    }

    return C;
  }
  KOKKOS_INLINE_FUNCTION
  static void print_Matrix(Matrix A) {
    for (int i = 0; i < npop; i++) {
      for (int j = 0; j < npop; j++) {
        std::cout << "A[" << i << "][" << j << "] = " << A[i][j] << std::endl;
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  static void print_vector(FState X) {
    for (int i = 0; i < npop; i++) {
      std::cout << "X[" << i << "] = " << X[i] << std::endl;
    }
  }

  KOKKOS_INLINE_FUNCTION
  static void print_vector(const FState &X) {
    for (int i = 0; i < npop; i++) {
      std::cout << "X[" << i << "] = " << X[i] << std::endl;
    }
  }
};

#endif
