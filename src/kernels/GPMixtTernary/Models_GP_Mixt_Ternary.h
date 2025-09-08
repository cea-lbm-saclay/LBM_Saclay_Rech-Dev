#ifndef MODELS_GP_MIXT_TERNARY_H_
#define MODELS_GP_MIXT_TERNARY_H_

#include <iostream>

#include <kokkos_shared.h>
#include <real_type.h>

#include "../Collision_operators.h"
#include "../vector_types.h"
#include "ConfigMap.h"
#include "LBMParams.h"
#include "LBM_maps.h"
#include "InitConditionsTypes.h"
#include "LBMParams.h"

#include "Index.h"

namespace PBM_GP_MIXT_TERNARY
{

using LBMState = typename Kokkos::Array<real_t, COMPONENT_SIZE>;
static constexpr real_t NORMAL_EPSILON = 1.0e-16;
    
// ================================================
//
// Models for mixt formulation of grand potential problems
//
// ================================================
struct ModelParams
{



    void showParams();

    ModelParams(){};
    ModelParams(const ConfigMap& configMap,const LBMParams& params);

    //! model params
    real_t epsiL, epsiS, msA, msB, mlA, mlB, Q;
    real_t DA1, DA0, DB1, DB0;
    real_t W, Mphi, lambda, gammaA, gammaB, counter_term, at_current, diffusion_current, diffusion_current2, dx, dt, e2, time;
    real_t elA, elB, elC, esA, esB, esC;
    //! init params
    real_t initCsA, initClA, initCsB, initClB;
    real_t x0, y0, z0, r0, x2, y2, z2, r2, sign, t0, freq;

    //! for init perco
    real_t initCsAA, initCsAB, initCsBA, initCsBB;

    //! for connected components methods
    real_t CC_phi_threshold;
    bool use_connected_components;
    bool print_cc_trace;
    bool apply_virtual_volume;
    real_t virtual_volume;
    // int virtual_volume_application_bdy;
    int virtual_volume_anchor_x;
    int virtual_volume_anchor_y;
    int virtual_volume_anchor_z;

    // should init at 0
    int cclabel_connected_to_virtual_volume;
    real_t cA_in_virtual_volume;
    real_t cB_in_virtual_volume;

    real_t A1A, B1A, A0A, B0A, A1B, B1B, A0B, B0B, xi, init_time;

    int initType;
    bool use_sharp_init;

    bool read_data_phi;
    std::string file_data_phi;
    bool read_data_cA;
    bool read_data_cB;

    KOKKOS_INLINE_FUNCTION real_t phi0(real_t x) const { return 0.5 * (1 + tanh(sign * 2.0 * x / W)); }

    // interpolation functions
    KOKKOS_INLINE_FUNCTION real_t p(real_t phi) const { return SQR(phi) * (3.0 - 2.0 * phi); }
    KOKKOS_INLINE_FUNCTION real_t p_prime(real_t phi) const { return 6.0 * phi * (1.0 - phi); }
    KOKKOS_INLINE_FUNCTION real_t g(real_t phi) const { return 8.0 * phi * phi * (1.0 - phi) * (1.0 - phi); }
    KOKKOS_INLINE_FUNCTION real_t g_prime(real_t phi) const { return 16.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi); }
    KOKKOS_INLINE_FUNCTION real_t h(real_t phi) const { return phi; }
    KOKKOS_INLINE_FUNCTION real_t h_prime(real_t phi) const { return 1.0; }
    KOKKOS_INLINE_FUNCTION real_t q(real_t phi) const { return phi; }
    KOKKOS_INLINE_FUNCTION real_t q_prime(real_t phi) const { return 1.0; }
    KOKKOS_INLINE_FUNCTION real_t aa(real_t phi) const { return phi * (3.0 - 2.0 * phi); }

    KOKKOS_INLINE_FUNCTION
    real_t m_norm(RVect2&, real_t Vx, real_t Vy, real_t Vz) const
    {
        return (sqrt(SQR(Vx) + SQR(Vy)) + NORMAL_EPSILON);
    }
    KOKKOS_INLINE_FUNCTION
    real_t m_norm(RVect3&, real_t Vx, real_t Vy, real_t Vz) const
    {
        return (sqrt(SQR(Vx) + SQR(Vy) + SQR(Vz)) + NORMAL_EPSILON);
    }

    // ================================================
    //
    // functions for the case of 2 quadratic free energies
    //
    // ================================================
    struct Tag2Quadra
    {
    };

    // =======================================================
    // compute mu A
    KOKKOS_INLINE_FUNCTION
    real_t compute_muA(Tag2Quadra tag, real_t phi, real_t CA) const
    {
        const real_t hv = h(phi);
        const real_t mu = CA - mlA * hv - msA * (1.0 - hv);
        return mu;
    }
    KOKKOS_INLINE_FUNCTION
    real_t compute_clA(Tag2Quadra tag, real_t mu) const
    {
        const real_t cl = (mu + mlA);
        return cl;
    }
    KOKKOS_INLINE_FUNCTION
    real_t compute_csA(Tag2Quadra tag, real_t mu) const
    {
        const real_t cs = (mu + msA);
        return cs;
    }
    // =======================================================
    // compute mu B
    KOKKOS_INLINE_FUNCTION
    real_t compute_muB(Tag2Quadra tag, real_t phi, real_t CB) const
    {
        const real_t hv = h(phi);
        const real_t mu = CB - mlB * hv - msB * (1.0 - hv);
        return mu;
    }

    KOKKOS_INLINE_FUNCTION
    real_t compute_clB(Tag2Quadra tag, real_t mu) const
    {
        const real_t cl = (mu + mlB);
        return cl;
    }
    KOKKOS_INLINE_FUNCTION
    real_t compute_csB(Tag2Quadra tag, real_t mu) const
    {
        const real_t cs = (mu + msB);
        return cs;
    }

    // =======================================================
    // compute delta w (from phi source term)
    KOKKOS_INLINE_FUNCTION
    real_t compute_dw(Tag2Quadra tag, LBMState& lbmState) const
    {
        real_t source_dw = (lbmState[IMU] * (msA - mlA) + lbmState[IMUB] * (msB - mlB) + Q);
        return source_dw;
    }
    // =======================================================
    // compute grand potential
    KOKKOS_INLINE_FUNCTION
    real_t compute_gp(Tag2Quadra tag, LBMState& lbmState) const
    {
        const real_t muA = lbmState[IMU];
        const real_t muB = lbmState[IMUB];
        const real_t phi = lbmState[IPHI];
        RVect<2> term;
        const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t gp = W * W * norm * norm + g(phi) + lambda * (p(phi) * (muA * mlA + muB * mlB - 0.5 * muA * muA - 0.5 * muB * muB + Q) + (1 - p(phi)) * (muA * msA + muB * msB - 0.5 * muA * muA - 0.5 * muB * muB));
        return gp;
    }
    // =======================================================
    // compute free energy
    KOKKOS_INLINE_FUNCTION
    real_t compute_free_energy(Tag2Quadra tag, LBMState& lbmState) const
    {
        //~ const real_t muA=lbmState[IMU];
        //~ const real_t muB=lbmState[IMUB];
        //~ const real_t phi=lbmState[IPHI];
        //~ RVect<2> term;
        //~ const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t fe = 0;
        return fe;
    }

    // =======================================================
    // =======================================================
    //
    //       equation 1 : phase field
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2Quadra type, EquationTag1 tag, LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2Quadra type, EquationTag1 tag, LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }
    // =======================================================
    // source term phi
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2Quadra type, EquationTag1 tag, LBMState& lbmState) const
    {
        real_t source_std = -lambda * Mphi / SQR(W) * p_prime(lbmState[IPHI]) * (lbmState[IMU] * (msA - mlA) + lbmState[IMUB] * (msB - mlB) + Q);
        real_t source_noct = -(1.0 - counter_term) * Mphi / SQR(W) * g_prime(lbmState[IPHI]);
        return (source_std + source_noct);
    }

    // =======================================================
    // advection terms for phase field
    template<int dim>
    KOKKOS_INLINE_FUNCTION
      RVect<dim>
      S1(Tag2Quadra type, EquationTag1 tag, LBMState& lbmState) const
    {
        RVect<dim> term;
        const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t force_ct = Mphi * counter_term * 4.0 * lbmState[IPHI] * (1.0 - lbmState[IPHI]) / W / norm;
        term[IX] = force_ct * lbmState[IDPHIDX];
        term[IY] = force_ct * lbmState[IDPHIDY];
        if (dim == 3)
        {
            term[IZ] = force_ct * lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of phase field equation
    KOKKOS_INLINE_FUNCTION
    real_t tau(Tag2Quadra type, EquationTag1 tag, LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 * Mphi * dt / SQR(dx));
        return (tau);
    }

    // =======================================================
    // =======================================================
    //
    //       equation 2 : composition
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2Quadra type, EquationTag2 tag, LBMState& lbmState) const
    {
        return lbmState[IC];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2Quadra type, EquationTag2 tag, LBMState& lbmState) const
    {

        const real_t M = gammaA * (lbmState[IMU]);
        return M;
    }

    // =======================================================
    // source term C
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2Quadra type, EquationTag2 tag, LBMState& lbmState) const
    {
        return 0.0;
    }
    // =======================================================
    // advection terms for diffusion equation
    template<int dim>
    KOKKOS_INLINE_FUNCTION
      RVect<dim>
      S1(Tag2Quadra type, EquationTag2 tag, LBMState& lbmState) const
    {
        //~ const real_t norm = sqrt( SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY])+ SQR(lbmState[IDPHIDZ]))+NORMAL_EPSILON;
        RVect<dim> term;
        const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t force_lbm = 0 * diffusion_current * (DA1 - DA0) * (lbmState[IMU]);
        real_t force_at = at_current * W * (1.0 - DA0 / DA1) / 4 * (mlA - msA) * lbmState[IDPHIDT] / norm;

        term[IX] = (force_lbm + force_at) * lbmState[IDPHIDX];
        term[IY] = (force_lbm + force_at) * lbmState[IDPHIDY];
        if (dim == 3)
        {
            term[IZ] = (force_lbm + force_at) * lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t tau(Tag2Quadra type, EquationTag2 tag, LBMState& lbmState) const
    {
        const real_t qv = q(lbmState[IPHI]);
        real_t tau = 0.5 + (qv * DA1 + (1 - qv) * DA0) * (e2 / gammaA * dt / SQR(dx));
        return (tau);
    }

    // =======================================================
    // =======================================================
    //
    //       equation 3 : composition B
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2Quadra type, EquationTag3 tag, LBMState& lbmState) const
    {
        return lbmState[ICB];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2Quadra type, EquationTag3 tag, LBMState& lbmState) const
    {

        const real_t M = gammaB * (lbmState[IMUB]);
        return M;
    }

    // =======================================================
    // source term C
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2Quadra type, EquationTag3 tag, LBMState& lbmState) const
    {
        return 0.0;
    }
    // =======================================================
    // advection terms for diffusion equation
    template<int dim>
    KOKKOS_INLINE_FUNCTION
      RVect<dim>
      S1(Tag2Quadra type, EquationTag3 tag, LBMState& lbmState) const
    {
        RVect<dim> term;
        const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t force_lbm = 0 * diffusion_current * (DB1 - DB0) * (lbmState[IMUB]);
        real_t force_at = at_current * W * (1.0 - DB0 / DB1) / 4 * (mlB - msB) * lbmState[IDPHIDT] / norm;
        term[IX] = (force_lbm + force_at) * lbmState[IDPHIDX];
        term[IY] = (force_lbm + force_at) * lbmState[IDPHIDY];
        if (dim == 3)
        {
            term[IZ] = (force_lbm + force_at) * lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t tau(Tag2Quadra type, EquationTag3 tag, LBMState& lbmState) const
    {
        const real_t qv = q(lbmState[IPHI]);
        real_t tau = 0.5 + (qv * DB1 + (1 - qv) * DB0) * (e2 / gammaB * dt / SQR(dx));
        return (tau);
    }

    // ================================================
    //
    // functions for the case of 2 quadratic free energies
    //
    // ================================================
    struct Tag2QuadraSolid
    {
    };

    // =======================================================
    // compute mu A
    KOKKOS_INLINE_FUNCTION
    real_t compute_muA(Tag2QuadraSolid tag, real_t phi, real_t CA) const
    {
        const real_t hv = h(phi);
        const real_t mu = CA - mlA * hv - msA * (1.0 - hv);
        return mu;
    }
    KOKKOS_INLINE_FUNCTION
    real_t compute_clA(Tag2QuadraSolid tag, real_t mu) const
    {
        const real_t cl = (mu + mlA);
        return cl;
    }
    KOKKOS_INLINE_FUNCTION
    real_t compute_csA(Tag2QuadraSolid tag, real_t mu) const
    {
        const real_t cs = (mu + msA);
        return cs;
    }
    // =======================================================
    // compute mu B
    KOKKOS_INLINE_FUNCTION
    real_t compute_muB(Tag2QuadraSolid tag, real_t phi, real_t CB) const
    {
        const real_t hv = h(phi);
        const real_t mu = CB - mlB * hv - msB * (1.0 - hv);
        return mu;
    }

    KOKKOS_INLINE_FUNCTION
    real_t compute_clB(Tag2QuadraSolid tag, real_t mu) const
    {
        const real_t cl = (mu + mlB);
        return cl;
    }
    KOKKOS_INLINE_FUNCTION
    real_t compute_csB(Tag2QuadraSolid tag, real_t mu) const
    {
        const real_t cs = (mu + msB);
        return cs;
    }
    // =======================================================
    // compute delta w (from phi source term)
    KOKKOS_INLINE_FUNCTION
    real_t compute_dw(Tag2QuadraSolid tag, LBMState& lbmState) const
    {
        real_t source_dw = -lambda * Mphi / SQR(W) * p_prime(lbmState[IPHI]) * (lbmState[IMU] * (msA - mlA) + lbmState[IMUB] * (msB - mlB) + Q);
        return source_dw;
    }

    // =======================================================
    // compute grand potential
    KOKKOS_INLINE_FUNCTION
    real_t compute_gp(Tag2QuadraSolid tag, LBMState& lbmState) const
    {
        //~ real_t muA=lbmState[IMU];
        //~ real_t muB=lbmState[IMUB];
        //~ real_t phi=lbmState[IPHI];

        return 0;
    }

    // =======================================================
    // compute free energy
    KOKKOS_INLINE_FUNCTION
    real_t compute_free_energy(Tag2QuadraSolid tag, LBMState& lbmState) const
    {
        //~ const real_t muA=lbmState[IMU];
        //~ const real_t muB=lbmState[IMUB];
        //~ const real_t phi=lbmState[IPHI];
        //~ RVect<2> term;
        //~ const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t fe = 0;
        return fe;
    }
    // =======================================================
    // =======================================================
    //
    //       equation 1 : phase field
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2QuadraSolid type, EquationTag1 tag, LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2QuadraSolid type, EquationTag1 tag, LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }
    // =======================================================
    // source term phi
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2QuadraSolid type, EquationTag1 tag, LBMState& lbmState) const
    {
        real_t source_std = -lambda * Mphi / SQR(W) * p_prime(lbmState[IPHI]) * (lbmState[IMU] * (msA - mlA) + lbmState[IMUB] * (msB - mlB) + Q);
        real_t source_noct = -(1.0 - counter_term) * Mphi / SQR(W) * g_prime(lbmState[IPHI]);
        return (source_std + source_noct);
    }

    // =======================================================
    // advection terms for phase field
    template<int dim>
    KOKKOS_INLINE_FUNCTION
      RVect<dim>
      S1(Tag2QuadraSolid type, EquationTag1 tag, LBMState& lbmState) const
    {
        RVect<dim> term;
        const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t force_ct = Mphi * counter_term * 4.0 * lbmState[IPHI] * (1.0 - lbmState[IPHI]) / W / norm;
        term[IX] = force_ct * lbmState[IDPHIDX];
        term[IY] = force_ct * lbmState[IDPHIDY];
        if (dim == 3)
        {
            term[IZ] = force_ct * lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of phase field equation
    KOKKOS_INLINE_FUNCTION
    real_t tau(Tag2QuadraSolid type, EquationTag1 tag, LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 * Mphi * dt / SQR(dx));
        return (tau);
    }

    // =======================================================
    // =======================================================
    //
    //       equation 2 : composition
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2QuadraSolid type, EquationTag2 tag, LBMState& lbmState) const
    {
        return lbmState[IC];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2QuadraSolid type, EquationTag2 tag, LBMState& lbmState) const
    {
        const real_t qv = q(lbmState[IPHI]);
        const real_t M = gammaA * (qv * DA1 + (1 - qv) * DA0) * (lbmState[IMU]);
        return M;
    }

    // =======================================================
    // source term C
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2QuadraSolid type, EquationTag2 tag, LBMState& lbmState) const
    {
        return 0.0;
    }
    // =======================================================
    // advection terms for diffusion equation
    template<int dim>
    KOKKOS_INLINE_FUNCTION
      RVect<dim>
      S1(Tag2QuadraSolid type, EquationTag2 tag, LBMState& lbmState) const
    {
        //~ const real_t norm = sqrt( SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY])+ SQR(lbmState[IDPHIDZ]))+NORMAL_EPSILON;
        RVect<dim> term;
        const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t force_lbm = diffusion_current * (DA1 - DA0) * (lbmState[IMU]);
        real_t force_at = at_current * W * (mlA - msA) * lbmState[IDPHIDT] / norm;

        term[IX] = (force_lbm + force_at) * lbmState[IDPHIDX];
        term[IY] = (force_lbm + force_at) * lbmState[IDPHIDY];
        if (dim == 3)
        {
            term[IZ] = (force_lbm + force_at) * lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t tau(Tag2QuadraSolid type, EquationTag2 tag, LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 / gammaA * dt / SQR(dx));
        return (tau);
    }

    // =======================================================
    // =======================================================
    //
    //       equation 3 : composition B
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2QuadraSolid type, EquationTag3 tag, LBMState& lbmState) const
    {
        return lbmState[ICB];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2QuadraSolid type, EquationTag3 tag, LBMState& lbmState) const
    {
        const real_t qv = q(lbmState[IPHI]);
        const real_t M = gammaB * (qv * DB1 + (1 - qv) * DB0) * (lbmState[IMUB]);
        return M;
    }

    // =======================================================
    // source term C
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2QuadraSolid type, EquationTag3 tag, LBMState& lbmState) const
    {
        return 0.0;
    }
    // =======================================================
    // advection terms for diffusion equation
    template<int dim>
    KOKKOS_INLINE_FUNCTION
      RVect<dim>
      S1(Tag2QuadraSolid type, EquationTag3 tag, LBMState& lbmState) const
    {
        RVect<dim> term;
        const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t force_lbm = diffusion_current * (DB1 - DB0) * (lbmState[IMUB]);
        real_t force_at = at_current * W * (mlB - msB) * lbmState[IDPHIDT] / norm;
        term[IX] = (force_lbm + force_at) * lbmState[IDPHIDX];
        term[IY] = (force_lbm + force_at) * lbmState[IDPHIDY];
        if (dim == 3)
        {
            term[IZ] = (force_lbm + force_at) * lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t tau(Tag2QuadraSolid type, EquationTag3 tag, LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 / gammaB * dt / SQR(dx));
        return (tau);
    }

    // ================================================
    //
    // functions for the case of 2 dilute free energies
    //
    // ================================================
    struct Tag2Dilute
    {
    };

    // =======================================================
    // compute mu A
    KOKKOS_INLINE_FUNCTION
    real_t compute_muA(Tag2Dilute tag, real_t phi, real_t CA) const
    {
        const real_t hv = h(phi);
        real_t mu;
        if (CA > 0)
        {
            mu = log(CA / (hv * exp(-elA) + (1 - hv) * exp(-esA)));
        }
        else
        {
            mu = -100;
        }
        return mu;
    }

    KOKKOS_INLINE_FUNCTION
    real_t compute_clA(Tag2Dilute tag, real_t mu) const
    {
        const real_t cl = exp(mu - elA);
        return cl;
    }
    KOKKOS_INLINE_FUNCTION
    real_t compute_csA(Tag2Dilute tag, real_t mu) const
    {
        const real_t cs = exp(mu - esA);
        return cs;
    }
    // =======================================================
    // compute mu B
    KOKKOS_INLINE_FUNCTION
    real_t compute_muB(Tag2Dilute tag, real_t phi, real_t CB) const
    {
        const real_t hv = h(phi);
        real_t mu;
        if (CB > 0)
        {
            mu = log((CB) / (hv * exp(-elB) + (1 - hv) * exp(-esB)));
        }
        else
        {
            mu = -100;
        }
        return mu;
    }

    KOKKOS_INLINE_FUNCTION
    real_t compute_clB(Tag2Dilute tag, real_t mu) const
    {
        const real_t cl = exp(mu - elB);
        return cl;
    }
    KOKKOS_INLINE_FUNCTION
    real_t compute_csB(Tag2Dilute tag, real_t mu) const
    {
        const real_t cs = exp(mu - esB);
        return cs;
    }

    // =======================================================
    // compute delta w (from phi source term)
    KOKKOS_INLINE_FUNCTION
    real_t compute_dw(Tag2Dilute tag, LBMState& lbmState) const
    {
        real_t dca = (exp(-elA) - exp(-esA));
        real_t dcb = (exp(-elB) - exp(-esB));
        real_t dw = (elC - esC - exp(lbmState[IMU]) * dca - exp(lbmState[IMUB]) * dcb);
        return dw;
    }

    // =======================================================
    // compute grand potential
    KOKKOS_INLINE_FUNCTION
    real_t compute_gp(Tag2Dilute tag, LBMState& lbmState) const
    {
        const real_t muA = lbmState[IMU];
        const real_t muB = lbmState[IMUB];
        const real_t phi = lbmState[IPHI];
        //~ RVect<2> term;
        //~ const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t gp = lambda * (p(phi) * (elC - exp(muA - elA) - exp(muB - elB)) + (1 - p(phi)) * (esC - exp(muA - esA) - exp(muB - esB)));
        return gp;
    }

    // =======================================================
    // compute free energy
    KOKKOS_INLINE_FUNCTION
    real_t compute_free_energy(Tag2Dilute tag, LBMState& lbmState) const
    {
        const real_t muA = lbmState[IMU];
        const real_t muB = lbmState[IMUB];
        const real_t phi = lbmState[IPHI];
        //~ RVect<2> term;
        //~ const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t fe = lambda * (p(phi) * (elC + exp(muA - elA) * (muA - 1) + exp(muB - elB) * (muB - 1)) + (1 - p(phi)) * (esC + exp(muA - esA) * (muA - 1) + exp(muB - esB) * (muB - 1)));
        return fe;
    }
    // =======================================================
    // =======================================================
    //
    //       equation 1 : phase field
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2Dilute type, EquationTag1 tag, const LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2Dilute type, EquationTag1 tag, const LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }
    // =======================================================
    // source term phi
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2Dilute type, EquationTag1 tag, const LBMState& lbmState) const
    {
        real_t dca = (exp(-elA) - exp(-esA));
        real_t dcb = (exp(-elB) - exp(-esB));
        real_t source_std = -lambda * Mphi / SQR(W) * p_prime(lbmState[IPHI]) * (elC - esC - exp(lbmState[IMU]) * dca - exp(lbmState[IMUB]) * dcb);
        real_t source_noct = -(1.0 - counter_term) * Mphi / SQR(W) * g_prime(lbmState[IPHI]);
        return (source_std + source_noct);
    }

    // =======================================================
    // advection terms for phase field
    template<int dim>
    KOKKOS_INLINE_FUNCTION
      RVect<dim>
      S1(Tag2Dilute type, EquationTag1 tag, const LBMState& lbmState) const
    {
        RVect<dim> term;
        const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t force_ct = Mphi * counter_term * 4.0 * lbmState[IPHI] * (1.0 - lbmState[IPHI]) / W / norm;
        term[IX] = force_ct * lbmState[IDPHIDX];
        term[IY] = force_ct * lbmState[IDPHIDY];
        if (dim == 3)
        {
            term[IZ] = force_ct * lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of phase field equation
    KOKKOS_INLINE_FUNCTION
    real_t tau(Tag2Dilute type, EquationTag1 tag, const LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 * Mphi * dt / SQR(dx));
        return (tau);
    }

    // =======================================================
    // =======================================================
    //
    //       equation 2 : composition
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2Dilute type, EquationTag2 tag, const LBMState& lbmState) const
    {
        return lbmState[IC];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2Dilute type, EquationTag2 tag, const LBMState& lbmState) const
    {

        const real_t M = gammaA * (lbmState[IMU]);
        return M;
    }

    // =======================================================
    // source term C
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2Dilute type, EquationTag2 tag, const LBMState& lbmState) const
    {
        return 0.0;
    }
    // =======================================================
    // advection terms for diffusion equation
    template<int dim>
    KOKKOS_INLINE_FUNCTION
      RVect<dim>
      S1(Tag2Dilute type, EquationTag2 tag, const LBMState& lbmState) const
    {

        RVect<dim> term;

        const real_t phi = lbmState[IPHI];
        const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t Cl = exp(lbmState[IMU] - elA);
        real_t Cs = exp(lbmState[IMU] - esA);
        real_t U = 0.0;
        if (lbmState[IPHI] < 0.5)
        {
            U = (Cl - Cs) * (1 - (phi * (1 - DA0 / DA1) + DA0 / DA1) * lbmState[IC]) / (4 * (1 - phi));
        }
        else
        {
            U = (Cl - Cs) * ((DA0 / DA1 * lbmState[IC] + phi * Cs) * (1 - 1 / Cl - phi) + phi * phi * Cl + phi) / (4 * (phi));
        }

        real_t force_at = at_current * W * U * lbmState[IDPHIDT] / norm;

        term[IX] = (force_at)*lbmState[IDPHIDX];
        term[IY] = (force_at)*lbmState[IDPHIDY];
        if (dim == 3)
        {
            term[IZ] = (force_at)*lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation for CA
    KOKKOS_INLINE_FUNCTION
    real_t tau(Tag2Dilute type, EquationTag2 tag, const LBMState& lbmState) const
    {
        const real_t qv = q(lbmState[IPHI]);
        real_t tau = 0.5 + (e2 * (qv * DA1 + (1 - qv) * DA0) * lbmState[IC] * dt / SQR(dx) / gammaA);
        return (tau);
    }

    // =======================================================
    // =======================================================
    //
    //       equation 3 : composition B
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2Dilute type, EquationTag3 tag, const LBMState& lbmState) const
    {
        return lbmState[ICB];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2Dilute type, EquationTag3 tag, const LBMState& lbmState) const
    {

        const real_t M = gammaB * (lbmState[IMUB]);
        return M;
    }

    // =======================================================
    // source term C
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2Dilute type, EquationTag3 tag, const LBMState& lbmState) const
    {
        return 0.0;
    }
    // =======================================================
    // advection terms for diffusion equation of CB
    template<int dim>
    KOKKOS_INLINE_FUNCTION
      RVect<dim>
      S1(Tag2Dilute type, EquationTag3 tag, const LBMState& lbmState) const
    {

        RVect<dim> term;

        const real_t phi = lbmState[IPHI];
        const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t Cl = exp(lbmState[IMUB] - elB);
        real_t Cs = exp(lbmState[IMUB] - esB);
        real_t U = 0.0;
        if (lbmState[IPHI] < 0.5)
        {
            U = (Cl - Cs) * (1 - (phi * (1 - DB0 / DB1) + DB0 / DB1) * lbmState[ICB]) / (4 * (1 - phi));
        }
        else
        {
            U = (Cl - Cs) * ((DB0 / DB1 * lbmState[ICB] + phi * Cs) * (1 - 1 / Cl - phi) + phi * phi * Cl + phi) / (4 * (phi));
        }

        real_t force_at = at_current * W * U * lbmState[IDPHIDT] / norm;

        term[IX] = (force_at)*lbmState[IDPHIDX];
        term[IY] = (force_at)*lbmState[IDPHIDY];
        if (dim == 3)
        {
            term[IZ] = (force_at)*lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t tau(Tag2Dilute type, EquationTag3 tag, const LBMState& lbmState) const
    {
        const real_t qv = q(lbmState[IPHI]);
        real_t tau = 0.5 + (e2 * (qv * DB1 + (1 - qv) * DB0) * lbmState[ICB] * dt / SQR(dx) / gammaB);
        return (tau);
    }

    // ================================================
    //
    // functions for the case of 2 dilute free energies with possibilty of D=0
    //
    // ================================================
    struct Tag2DiluteSolid
    {
    };

    // =======================================================
    // compute mu A
    KOKKOS_INLINE_FUNCTION
    real_t compute_muA(Tag2DiluteSolid tag, real_t phi, real_t CA) const
    {
        const real_t hv = h(phi);
        real_t mu;
        if (CA > 0)
        {
            mu = log(CA / (hv * exp(-elA) + (1 - hv) * exp(-esA)));
        }
        else
        {
            mu = -100;
        }
        return mu;
    }
    KOKKOS_INLINE_FUNCTION
    real_t compute_clA(Tag2DiluteSolid tag, real_t mu) const
    {
        const real_t cl = exp(mu - elA);
        return cl;
    }
    KOKKOS_INLINE_FUNCTION
    real_t compute_csA(Tag2DiluteSolid tag, real_t mu) const
    {
        const real_t cs = exp(mu - esA);
        return cs;
    }
    // =======================================================
    // compute mu B
    KOKKOS_INLINE_FUNCTION
    real_t compute_muB(Tag2DiluteSolid tag, real_t phi, real_t CB) const
    {
        const real_t hv = h(phi);
        real_t mu;
        if (CB > 0)
        {
            mu = log((CB) / (hv * exp(-elB) + (1 - hv) * exp(-esB)));
        }
        else
        {
            mu = -100;
        }
        return mu;
    }

    KOKKOS_INLINE_FUNCTION
    real_t compute_clB(Tag2DiluteSolid tag, real_t mu) const
    {
        const real_t cl = exp(mu - elB);
        return cl;
    }
    KOKKOS_INLINE_FUNCTION
    real_t compute_csB(Tag2DiluteSolid tag, real_t mu) const
    {
        const real_t cs = exp(mu - esB);
        return cs;
    }
    // =======================================================
    // compute delta w (from phi source term)
    KOKKOS_INLINE_FUNCTION
    real_t compute_dw(Tag2DiluteSolid tag, const LBMState& lbmState) const
    {

        real_t dca = (exp(-elA) - exp(-esA));
        real_t dcb = (exp(-elB) - exp(-esB));
        real_t dw = (elC - esC - exp(lbmState[IMU]) * dca - exp(lbmState[IMUB]) * dcb);
        return dw;
    }

    // =======================================================
    // compute grand potential
    KOKKOS_INLINE_FUNCTION
    real_t compute_gp(Tag2DiluteSolid tag, const LBMState& lbmState) const
    {
        const real_t muA = lbmState[IMU];
        const real_t muB = lbmState[IMUB];
        const real_t phi = lbmState[IPHI];
        //~ RVect<2> term;
        //~ const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t gp = lambda * (p(phi) * (elC - exp(muA - elA) - exp(muB - elB)) + (1 - p(phi)) * (esC - exp(muA - esA) - exp(muB - esB)));
        return gp;
    }

    // =======================================================
    // compute free energy
    KOKKOS_INLINE_FUNCTION
    real_t compute_free_energy(Tag2DiluteSolid tag, LBMState& lbmState) const
    {
        const real_t muA = lbmState[IMU];
        const real_t muB = lbmState[IMUB];
        const real_t phi = lbmState[IPHI];
        //~ RVect<2> term;
        //~ const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t fe = lambda * (p(phi) * (elC + exp(muA - elA) * (muA - 1) + exp(muB - elB) * (muB - 1)) + (1 - p(phi)) * (esC + exp(muA - esA) * (muA - 1) + exp(muB - esB) * (muB - 1)));
        return fe;
    }

    // =======================================================
    // =======================================================
    //
    //       equation 1 : phase field
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2DiluteSolid type, EquationTag1 tag, const LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2DiluteSolid type, EquationTag1 tag, LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }
    // =======================================================
    // source term phi
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2DiluteSolid type, EquationTag1 tag, LBMState& lbmState) const
    {
        real_t dca = (exp(-elA) - exp(-esA));
        real_t dcb = (exp(-elB) - exp(-esB));
        real_t source_std = -lambda * Mphi / SQR(W) * p_prime(lbmState[IPHI]) * (elC - esC - exp(lbmState[IMU]) * dca - exp(lbmState[IMUB]) * dcb);
        real_t source_noct = -(1.0 - counter_term) * Mphi / SQR(W) * g_prime(lbmState[IPHI]);
        return (source_std + source_noct);
    }

    // =======================================================
    // advection terms for phase field
    template<int dim>
    KOKKOS_INLINE_FUNCTION
      RVect<dim>
      S1(Tag2DiluteSolid type, EquationTag1 tag, LBMState& lbmState) const
    {
        RVect<dim> term;
        const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t force_ct = Mphi * counter_term * 4.0 * lbmState[IPHI] * (1.0 - lbmState[IPHI]) / W / norm;
        term[IX] = force_ct * lbmState[IDPHIDX];
        term[IY] = force_ct * lbmState[IDPHIDY];
        if (dim == 3)
        {
            term[IZ] = force_ct * lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of phase field equation
    KOKKOS_INLINE_FUNCTION
    real_t tau(Tag2DiluteSolid type, EquationTag1 tag, LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 * Mphi * dt / SQR(dx));
        return (tau);
    }

    // =======================================================
    // =======================================================
    //
    //       equation 2 : composition
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2DiluteSolid type, EquationTag2 tag, LBMState& lbmState) const
    {
        return lbmState[IC];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2DiluteSolid type, EquationTag2 tag, LBMState& lbmState) const
    {
        const real_t qv = q(lbmState[IPHI]);
        const real_t M = gammaA * (qv * DA1 + (1 - qv) * DA0) * (lbmState[IMU]);
        return M;
    }

    // =======================================================
    // source term C
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2DiluteSolid type, EquationTag2 tag, LBMState& lbmState) const
    {
        return 0.0;
    }
    // =======================================================
    // advection terms for diffusion equation
    template<int dim>
    KOKKOS_INLINE_FUNCTION
      RVect<dim>
      S1(Tag2DiluteSolid type, EquationTag2 tag, LBMState& lbmState) const
    {
        //~ const real_t qv = q(lbmState[IPHI]);
        real_t force_lbm = diffusion_current * (DA1 - DA0) * (lbmState[IMU]) * lbmState[IC];

        real_t force_at = 0.0;

        RVect<dim> term;
        term[IX] = (force_lbm + force_at) * lbmState[IDPHIDX];
        term[IY] = (force_lbm + force_at) * lbmState[IDPHIDY];
        if (dim == 3)
        {
            term[IZ] = (force_lbm + force_at) * lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation for CA
    KOKKOS_INLINE_FUNCTION
    real_t tau(Tag2DiluteSolid type, EquationTag2 tag, LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 * lbmState[IC] * dt / SQR(dx) / gammaA);
        return (tau);
    }

    // =======================================================
    // =======================================================
    //
    //       equation 3 : composition B
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2DiluteSolid type, EquationTag3 tag, LBMState& lbmState) const
    {
        return lbmState[ICB];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2DiluteSolid type, EquationTag3 tag, LBMState& lbmState) const
    {
        const real_t qv = q(lbmState[IPHI]);
        const real_t M = gammaB * (qv * DB1 + (1 - qv) * DB0) * (lbmState[IMUB]);
        return M;
    }

    // =======================================================
    // source term C
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2DiluteSolid type, EquationTag3 tag, LBMState& lbmState) const
    {
        return 0.0;
    }
    // =======================================================
    // advection terms for diffusion equation
    template<int dim>
    KOKKOS_INLINE_FUNCTION
      RVect<dim>
      S1(Tag2DiluteSolid type, EquationTag3 tag, LBMState& lbmState) const
    {
        //~ const real_t norm = sqrt( SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY])+ SQR(lbmState[IDPHIDZ]))+NORMAL_EPSILON;
        //~ const real_t qv = q(lbmState[IPHI]);
        real_t force_lbm = diffusion_current * (DB1 - DB0) * lbmState[IMUB] * lbmState[ICB];

        real_t force_at = 0.0;

        RVect<dim> term;
        term[IX] = (force_lbm + force_at) * lbmState[IDPHIDX];
        term[IY] = (force_lbm + force_at) * lbmState[IDPHIDY];
        if (dim == 3)
        {
            term[IZ] = (force_lbm + force_at) * lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t tau(Tag2DiluteSolid type, EquationTag3 tag, LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 * lbmState[ICB] * dt / SQR(dx) / gammaB);
        return (tau);
    }

    // ================================================
    //
    // functions for the case of 2 dilute free energies with instant diffusion in liquid. relaxation rate needs to be adjusted by tuning Dl, which does not
    //  matter too much because diffusion in liquid is resolved a posteriori by taking the mean of the composition in the liquid (or pores).
    // everything other than tauCA and tauCB stays the same
    //
    // ================================================
    struct Tag2DiluteInstantDiffusion
    {
    };

    // =======================================================
    // compute mu A
    KOKKOS_INLINE_FUNCTION
    real_t compute_muA(Tag2DiluteInstantDiffusion tag, real_t phi, real_t CA) const
    {
        const real_t hv = h(phi);
        real_t mu;
        if (CA > 0)
        {
            mu = log(CA / (hv * exp(-elA) + (1 - hv) * exp(-esA)));
        }
        else
        {
            mu = -100;
        }
        return mu;
    }

    KOKKOS_INLINE_FUNCTION
    real_t compute_clA(Tag2DiluteInstantDiffusion tag, real_t mu) const
    {
        const real_t cl = exp(mu - elA);
        return cl;
    }
    KOKKOS_INLINE_FUNCTION
    real_t compute_csA(Tag2DiluteInstantDiffusion tag, real_t mu) const
    {
        const real_t cs = exp(mu - esA);
        return cs;
    }
    // =======================================================
    // compute mu B
    KOKKOS_INLINE_FUNCTION
    real_t compute_muB(Tag2DiluteInstantDiffusion tag, real_t phi, real_t CB) const
    {
        const real_t hv = h(phi);
        real_t mu;
        if (CB > 0)
        {
            mu = log((CB) / (hv * exp(-elB) + (1 - hv) * exp(-esB)));
        }
        else
        {
            mu = -100;
        }
        return mu;
    }

    KOKKOS_INLINE_FUNCTION
    real_t compute_clB(Tag2DiluteInstantDiffusion tag, real_t mu) const
    {
        const real_t cl = exp(mu - elB);
        return cl;
    }
    KOKKOS_INLINE_FUNCTION
    real_t compute_csB(Tag2DiluteInstantDiffusion tag, real_t mu) const
    {
        const real_t cs = exp(mu - esB);
        return cs;
    }

    // =======================================================
    // compute delta w (from phi source term)
    KOKKOS_INLINE_FUNCTION
    real_t compute_dw(Tag2DiluteInstantDiffusion tag, LBMState& lbmState) const
    {
        real_t dca = (exp(-elA) - exp(-esA));
        real_t dcb = (exp(-elB) - exp(-esB));
        real_t dw = (elC - esC - exp(lbmState[IMU]) * dca - exp(lbmState[IMUB]) * dcb);
        return dw;
    }

    // =======================================================
    // compute grand potential
    KOKKOS_INLINE_FUNCTION
    real_t compute_gp(Tag2DiluteInstantDiffusion tag, LBMState& lbmState) const
    {
        const real_t muA = lbmState[IMU];
        const real_t muB = lbmState[IMUB];
        const real_t phi = lbmState[IPHI];
        //~ RVect<2> term;
        //~ const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t gp = lambda * (p(phi) * (elC - exp(muA - elA) - exp(muB - elB)) + (1 - p(phi)) * (esC - exp(muA - esA) - exp(muB - esB)));
        return gp;
    }

    // =======================================================
    // compute free energy
    KOKKOS_INLINE_FUNCTION
    real_t compute_free_energy(Tag2DiluteInstantDiffusion tag, LBMState& lbmState) const
    {
        const real_t muA = lbmState[IMU];
        const real_t muB = lbmState[IMUB];
        const real_t phi = lbmState[IPHI];
        //~ RVect<2> term;
        //~ const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t fe = lambda * (p(phi) * (elC + exp(muA - elA) * (muA - 1) + exp(muB - elB) * (muB - 1)) + (1 - p(phi)) * (esC + exp(muA - esA) * (muA - 1) + exp(muB - esB) * (muB - 1)));
        return fe;
    }
    // =======================================================
    // =======================================================
    //
    //       equation 1 : phase field
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2DiluteInstantDiffusion type, EquationTag1 tag, const LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2DiluteInstantDiffusion type, EquationTag1 tag, const LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }
    // =======================================================
    // source term phi
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2DiluteInstantDiffusion type, EquationTag1 tag, const LBMState& lbmState) const
    {
        real_t dca = (exp(-elA) - exp(-esA));
        real_t dcb = (exp(-elB) - exp(-esB));
        real_t source_std = -lambda * Mphi / SQR(W) * p_prime(lbmState[IPHI]) * (elC - esC - exp(lbmState[IMU]) * dca - exp(lbmState[IMUB]) * dcb);
        real_t source_noct = -(1.0 - counter_term) * Mphi / SQR(W) * g_prime(lbmState[IPHI]);
        return (source_std + source_noct);
    }

    // =======================================================
    // advection terms for phase field
    template<int dim>
    KOKKOS_INLINE_FUNCTION
      RVect<dim>
      S1(Tag2DiluteInstantDiffusion type, EquationTag1 tag, const LBMState& lbmState) const
    {
        RVect<dim> term;
        const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t force_ct = Mphi * counter_term * 4.0 * lbmState[IPHI] * (1.0 - lbmState[IPHI]) / W / norm;
        term[IX] = force_ct * lbmState[IDPHIDX];
        term[IY] = force_ct * lbmState[IDPHIDY];
        if (dim == 3)
        {
            term[IZ] = force_ct * lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of phase field equation
    KOKKOS_INLINE_FUNCTION
    real_t tau(Tag2DiluteInstantDiffusion type, EquationTag1 tag, const LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 * Mphi * dt / SQR(dx));
        return (tau);
    }

    // =======================================================
    // =======================================================
    //
    //       equation 2 : composition
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2DiluteInstantDiffusion type, EquationTag2 tag, const LBMState& lbmState) const
    {
        return lbmState[IC];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2DiluteInstantDiffusion type, EquationTag2 tag, const LBMState& lbmState) const
    {

        const real_t M = gammaA * (lbmState[IMU]);
        return M;
    }

    // =======================================================
    // source term C
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2DiluteInstantDiffusion type, EquationTag2 tag, const LBMState& lbmState) const
    {
        return 0.0;
    }
    // =======================================================
    // advection terms for diffusion equation
    template<int dim>
    KOKKOS_INLINE_FUNCTION
      RVect<dim>
      S1(Tag2DiluteInstantDiffusion type, EquationTag2 tag, const LBMState& lbmState) const
    {

        RVect<dim> term;

        const real_t phi = lbmState[IPHI];
        const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t Cl = exp(lbmState[IMU] - elA);
        real_t Cs = exp(lbmState[IMU] - esA);
        real_t U = 0.0;
        if (lbmState[IPHI] < 0.5)
        {
            U = (Cl - Cs) * (1 - (phi * (1 - DA0 / DA1) + DA0 / DA1) * lbmState[IC]) / (4 * (1 - phi));
        }
        else
        {
            U = (Cl - Cs) * ((DA0 / DA1 * lbmState[IC] + phi * Cs) * (1 - 1 / Cl - phi) + phi * phi * Cl + phi) / (4 * (phi));
        }

        real_t force_at = at_current * W * U * lbmState[IDPHIDT] / norm;

        term[IX] = (force_at)*lbmState[IDPHIDX];
        term[IY] = (force_at)*lbmState[IDPHIDY];
        if (dim == 3)
        {
            term[IZ] = (force_at)*lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation for CA
    KOKKOS_INLINE_FUNCTION
    real_t tau(Tag2DiluteInstantDiffusion type, EquationTag2 tag, const LBMState& lbmState) const
    {
        const real_t muA = lbmState[IMU];
        const real_t csA = compute_csA(type, muA);
        real_t tau = 0.5 + (e2 * (DA0 * csA) * dt / SQR(dx) / gammaA);
        return (tau);
    }

    // =======================================================
    // =======================================================
    //
    //       equation 3 : composition B
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2DiluteInstantDiffusion type, EquationTag3 tag, const LBMState& lbmState) const
    {
        return lbmState[ICB];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2DiluteInstantDiffusion type, EquationTag3 tag, const LBMState& lbmState) const
    {

        const real_t M = gammaB * (lbmState[IMUB]);
        return M;
    }

    // =======================================================
    // source term C
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2DiluteInstantDiffusion type, EquationTag3 tag, const LBMState& lbmState) const
    {
        return 0.0;
    }
    // =======================================================
    // advection terms for diffusion equation
    template<int dim>
    KOKKOS_INLINE_FUNCTION
      RVect<dim>
      S1(Tag2DiluteInstantDiffusion type, EquationTag3 tag, const LBMState& lbmState) const
    {

        RVect<dim> term;

        const real_t phi = lbmState[IPHI];
        const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t Cl = exp(lbmState[IMU] - elB);
        real_t Cs = exp(lbmState[IMU] - esB);
        real_t U = 0.0;
        if (lbmState[IPHI] < 0.5)
        {
            U = (Cl - Cs) * (1 - (phi * (1 - DB0 / DB1) + DB0 / DB1) * lbmState[IC]) / (4 * (1 - phi));
        }
        else
        {
            U = (Cl - Cs) * ((DB0 / DB1 * lbmState[IC] + phi * Cs) * (1 - 1 / Cl - phi) + phi * phi * Cl + phi) / (4 * (phi));
        }

        real_t force_at = at_current * W * U * lbmState[IDPHIDT] / norm;

        term[IX] = (force_at)*lbmState[IDPHIDX];
        term[IY] = (force_at)*lbmState[IDPHIDY];
        if (dim == 3)
        {
            term[IZ] = (force_at)*lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t tau(Tag2DiluteInstantDiffusion type, EquationTag3 tag, const LBMState& lbmState) const
    {
        const real_t muB = lbmState[IMUB];
        const real_t csB = compute_csB(type, muB);
        real_t tau = 0.5 + (e2 * (DB0 * csB) * dt / SQR(dx) / gammaB);
        return (tau);
    }
}; // end struct model
}
#endif // MODELS_DISSOL_GP_MIXT_H_
