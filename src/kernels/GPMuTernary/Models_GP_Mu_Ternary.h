#ifndef MODELS_GP_MU_TERNARY_H_
#define MODELS_GP_MU_TERNARY_H_
#include "InitConditionsTypes.h"
#include "GPMu_Ternary_Index.h"
#include <real_type.h>

namespace PBM_GP_MU_TERNARY {

// ================================================
//
// Models for mixt formulation of grand potential problems
//
// ================================================

struct ModelParams {

    using LBMState = typename Kokkos::Array<real_t, COMPONENT_SIZE>;
    static constexpr real_t NORMAL_EPSILON = 1.0e-16;

    void showParams()
    {
        std::cout << "W :    " << W << std::endl;
        std::cout << "counter_term :    " << counter_term << std::endl;
        std::cout << "at_current :    " << at_current << std::endl;
        std::cout << "init :" << initType << "." << std::endl;
        std::cout << "initClA :" << initClA << "." << std::endl;
        std::cout << "initCsA :" << initCsA << "." << std::endl;
        std::cout << "initClB :" << initClB << "." << std::endl;
        std::cout << "initCsB :" << initCsB << "." << std::endl;
        //~ real_t tau = 0.5 + (e2 * Mphi * dt / SQR(dx));
        std::cout << "tauPhi :" << (0.5 + (e2 * Mphi * dt / SQR(dx))) << std::endl;
        std::cout << "tauCA :" << (0.5 + (e2 / gammaA * dt / SQR(dx))) << std::endl;
        std::cout << "tauCB :" << (0.5 + (e2 / gammaB * dt / SQR(dx))) << std::endl;
    };
    ModelParams() {};
    ModelParams(const ConfigMap& configMap, LBMParams params)
    {
        dt = params.dt;
        dx = params.dx;
        e2 = configMap.getFloat("lbm", "e2", 3.0);
        W = configMap.getFloat("params", "W", 0.005);
        Mphi = configMap.getFloat("params", "Mphi", 1.2);
        lambda = configMap.getFloat("params", "lambda", 230.0);

        epsiL = configMap.getFloat("params", "epsiL", 1.0);
        epsiS = configMap.getFloat("params", "epsiS", 1.0);
        mlA = configMap.getFloat("params", "mlA", 0.1);
        mlB = configMap.getFloat("params", "mlB", 0.1);
        msA = configMap.getFloat("params", "msA", 0.2);
        msB = configMap.getFloat("params", "msB", 0.2);
        Q = configMap.getFloat("params", "Q", -0.04);

        DA1 = configMap.getFloat("params", "DA1", 1.0);
        DA0 = configMap.getFloat("params", "DA0", 1.0);
        DB1 = configMap.getFloat("params", "DB1", 1.0);
        DB0 = configMap.getFloat("params", "DB0", 1.0);

        gammaA = configMap.getFloat("params", "gammaA", 1.0);
        gammaB = configMap.getFloat("params", "gammaB", 1.0);

        counter_term = configMap.getFloat("params", "counter_term", 0.0);
        at_current = configMap.getFloat("params", "at_current", 0.0);
        diffusion_current = configMap.getFloat("params", "diffusion_current", 1.0);
        diffusion_current2 = configMap.getBool("params", "diffusion_current2", false);

        elA = configMap.getFloat("params", "elA", 0.1);
        esA = configMap.getFloat("params", "esA", 0.1);
        elB = configMap.getFloat("params", "elB", 0.1);
        esB = configMap.getFloat("params", "esB", 0.1);
        elC = configMap.getFloat("params", "elC", 0.1);
        esC = configMap.getFloat("params", "esC", 0.1);

        x0 = configMap.getFloat("init", "x0", 0.0);
        y0 = configMap.getFloat("init", "y0", 0.0);
        z0 = configMap.getFloat("init", "z0", 0.0);
        r0 = configMap.getFloat("init", "r0", 0.2);

        x2 = configMap.getFloat("init", "x2", 0.0);
        y2 = configMap.getFloat("init", "y2", 0.0);
        z2 = configMap.getFloat("init", "z2", 0.0);
        r2 = configMap.getFloat("init", "r2", 0.2);

        initClA = configMap.getFloat("init", "initClA", 0.1);
        initCsA = configMap.getFloat("init", "initCsA", 0.2);
        initClB = configMap.getFloat("init", "initClB", 0.1);
        initCsB = configMap.getFloat("init", "initCsB", 0.2);
        sign = configMap.getFloat("init", "sign", 1);
        t0 = configMap.getFloat("init", "t0", 1.0);
        initErfc = configMap.getFloat("init", "initErfc", 0.1);
        xi = configMap.getFloat("init", "xi", 0.0);
        freq = configMap.getFloat("init", "freq", 5);

        initType = PHASE_FIELD_INIT_UNDEFINED;
        std::string initTypeStr = std::string(configMap.getString("init", "init_type", "unknown"));

        if (initTypeStr == "vertical")
            initType = PHASE_FIELD_INIT_VERTICAL;
        else if (initTypeStr == "sphere")
            initType = PHASE_FIELD_INIT_SPHERE;
        else if (initTypeStr == "2sphere")
            initType = PHASE_FIELD_INIT_2SPHERE;
        else if (initTypeStr == "square")
            initType = PHASE_FIELD_INIT_SQUARE;
        else if (initTypeStr == "data")
            initType = PHASE_FIELD_INIT_DATA;
        else if (initTypeStr == "ternary_random")
            initType = PHASE_FIELD_RANDOM_TERNARY_GLASS;
        else if (initTypeStr == "ternary_random_sphere")
            initType = PHASE_FIELD_RANDOM_TERNARY_GLASS_SPHERE;

        showParams();
    }

    //! model params
    real_t epsiL, epsiS, msA, msB, mlA, mlB, Q;
    real_t DA1, DA0, DB1, DB0;
    real_t W, Mphi, lambda, gammaA, gammaB, counter_term, at_current, diffusion_current, diffusion_current2, dx, dt, e2;
    real_t elA, elB, elC, esA, esB, esC;
    //! init params
    real_t initCsA, initClA, initCsB, initClB;
    real_t x0, y0, z0, r0, x2, y2, z2, r2, sign, t0, initErfc, xi, freq;
    int initType;

    KOKKOS_INLINE_FUNCTION real_t phi0(real_t x) const { return 0.5 * (1 + tanh(sign * 2.0 * x / W)); }

    // interpolation functions
    KOKKOS_INLINE_FUNCTION real_t p(real_t phi) const { return SQR(phi) * (3.0 - 2.0 * phi); }
    KOKKOS_INLINE_FUNCTION real_t p_prime(real_t phi) const { return 6.0 * phi * (1.0 - phi); }
    KOKKOS_INLINE_FUNCTION real_t g_prime(real_t phi) const { return 16.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi); }
    KOKKOS_INLINE_FUNCTION real_t h(real_t phi) const { return phi; }
    KOKKOS_INLINE_FUNCTION real_t h_prime(real_t phi) const { return 1.0; }
    KOKKOS_INLINE_FUNCTION real_t q(real_t phi) const { return phi; }
    KOKKOS_INLINE_FUNCTION real_t q_prime(real_t phi) const { return 1.0; }
    KOKKOS_INLINE_FUNCTION real_t aa(real_t phi) const { return phi * (3.0 - 2.0 * phi); }

    // ================================================
    //
    // functions for the case of 2 quadratic free energies
    //
    // ================================================
    struct Tag2Quadra {
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
    // =======================================================
    // compute mu B
    KOKKOS_INLINE_FUNCTION
    real_t compute_muB(Tag2Quadra tag, real_t phi, real_t CB) const
    {
        const real_t hv = h(phi);
        const real_t mu = CB - mlB * hv - msB * (1.0 - hv);
        return mu;
    }

    // =======================================================
    // compute C A
    KOKKOS_INLINE_FUNCTION
    real_t compute_CA(Tag2Quadra tag, real_t phi, real_t muA) const
    {
        const real_t hv = h(phi);
        const real_t CA = muA + mlA * hv + msA * (1.0 - hv);
        return CA;
    }
    // =======================================================
    // compute C B
    KOKKOS_INLINE_FUNCTION
    real_t compute_CB(Tag2Quadra tag, real_t phi, real_t muB) const
    {
        const real_t hv = h(phi);
        const real_t CB = muB + mlB * hv + msB * (1.0 - hv);
        return CB;
    }

    // =======================================================
    // source term phi
    KOKKOS_INLINE_FUNCTION
    real_t source_phi(Tag2Quadra tag, LBMState& lbmState) const
    {
        real_t source_std = -lambda * Mphi / SQR(W) * p_prime(lbmState[IPHI]) * (lbmState[IMU] * (msA - mlA) + lbmState[IMUB] * (msB - mlB) + Q);
        real_t source_noct = -(1.0 - counter_term) * Mphi / SQR(W) * g_prime(lbmState[IPHI]);
        return (source_std + source_noct);
    }

    // =======================================================
    // advection terms for phase field
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        force_phi(Tag2Quadra tag, LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY]) + SQR(lbmState[IDPHIDZ])) + NORMAL_EPSILON;
        real_t force_ct = dx / dt * counter_term * 4.0 * lbmState[IPHI] * (1.0 - lbmState[IPHI]) / W / norm;
        RVect<dim> term;
        term[IX] = force_ct * lbmState[IDPHIDX];
        term[IY] = force_ct * lbmState[IDPHIDY];
        if (dim == 3) {
            term[IZ] = force_ct * lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of phase field equation
    KOKKOS_INLINE_FUNCTION
    real_t tau_phi(Tag2Quadra tag, LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 * Mphi * dt / SQR(dx));
        return (tau);
    }

    // =======================================================
    // source term muA
    KOKKOS_INLINE_FUNCTION
    real_t source_A(Tag2Quadra tag, LBMState& lbmState) const
    {
        const real_t mu = lbmState[IMU];
        real_t source_std = h_prime(lbmState[IPHI]) * (compute_CA(tag, 1.0, mu) - compute_CA(tag, 0.0, mu)) * lbmState[IDPHIDT];
        return (source_std);
    }
    // =======================================================
    // advection terms for diffusion equation
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        force_CA(Tag2Quadra tag, LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY]) + SQR(lbmState[IDPHIDZ])) + NORMAL_EPSILON;
        real_t force_lbm = diffusion_current * dx / dt * gammaA * (DA1 - DA0) * (lbmState[IMU]);
        real_t force_at = dx / dt * gammaA * at_current * W * (mlA - msA) * lbmState[IDPHIDT] / norm;
        RVect<dim> term;
        term[IX] = (force_lbm + force_at) * lbmState[IDPHIDX];
        term[IY] = (force_lbm + force_at) * lbmState[IDPHIDY];
        if (dim == 3) {
            term[IZ] = (force_lbm + force_at) * lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t tau_CA(Tag2Quadra tag, LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 / gammaA * dt / SQR(dx));
        return (tau);
    }

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t kiA(Tag2Quadra tag, LBMState& lbmState) const
    {
        return (1.0);
    }

    // =======================================================
    // mobility of the diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t compute_mobilityA(Tag2Quadra tag, LBMState& lbmState) const
    {
        const real_t qv = q(lbmState[IPHI]);
        const real_t M = gammaA * (qv * DA1 + (1 - qv) * DA0);
        return M;
    }
    // =======================================================
    // source term muB
    KOKKOS_INLINE_FUNCTION
    real_t source_B(Tag2Quadra tag, LBMState& lbmState) const
    {
        const real_t mu = lbmState[IMUB];
        real_t source_std = h_prime(lbmState[IPHI]) * (compute_CB(tag, 1.0, mu) - compute_CB(tag, 0.0, mu)) * lbmState[IDPHIDT];
        return (source_std);
    }
    // =======================================================
    // advection terms for diffusion equation
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        force_CB(Tag2Quadra tag, LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY]) + SQR(lbmState[IDPHIDZ])) + NORMAL_EPSILON;
        real_t force_lbm = diffusion_current * dx / dt * gammaB * (DB1 - DB0) * (lbmState[IMUB]);
        real_t force_at = dx / dt * gammaB * at_current * W * (mlB - msB) * lbmState[IDPHIDT] / norm;
        RVect<dim> term;
        term[IX] = (force_lbm + force_at) * lbmState[IDPHIDX];
        term[IY] = (force_lbm + force_at) * lbmState[IDPHIDY];
        if (dim == 3) {
            term[IZ] = (force_lbm + force_at) * lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t tau_CB(Tag2Quadra tag, LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 / gammaB * dt / SQR(dx));
        return (tau);
    }

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t kiB(Tag2Quadra tag, LBMState& lbmState) const
    {
        return (1.0);
    }

    // =======================================================
    // mobility of the diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t compute_mobilityB(Tag2Quadra tag, LBMState& lbmState) const
    {
        const real_t qv = q(lbmState[IPHI]);
        const real_t M = gammaB * (qv * DB1 + (1 - qv) * DB0);
        return M;
    }

    // ================================================
    //
    // functions for the case of 2 dilute free energies
    //
    // ================================================
    struct Tag2Dilute {
    };

    // =======================================================
    // compute mu A
    KOKKOS_INLINE_FUNCTION
    real_t compute_muA(Tag2Dilute tag, real_t phi, real_t CA) const
    {
        const real_t hv = h(phi);
        real_t mu;
        if (CA > 0) {
            mu = log(CA / (hv * exp(-elA) + (1 - hv) * exp(-esA)));
        } else {
            mu = -100;
        }
        return mu;
    }
    // =======================================================
    // compute mu B
    KOKKOS_INLINE_FUNCTION
    real_t compute_muB(Tag2Dilute tag, real_t phi, real_t CB) const
    {
        const real_t hv = h(phi);
        real_t mu;
        if (CB > 0) {
            mu = log((CB) / (hv * exp(-elB) + (1 - hv) * exp(-esB)));
        } else {
            mu = -100;
        }
        return mu;
    }

    // =======================================================
    // compute C A
    KOKKOS_INLINE_FUNCTION
    real_t compute_CA(Tag2Dilute tag, real_t phi, real_t muA) const
    {
        const real_t hv = h(phi);
        real_t CA = ((hv * exp(-elA) + (1 - hv) * exp(-esA))) * exp(muA);
        return CA;
    }
    // =======================================================
    // compute C B
    KOKKOS_INLINE_FUNCTION
    real_t compute_CB(Tag2Dilute tag, real_t phi, real_t muB) const
    {
        const real_t hv = h(phi);
        real_t CB = ((hv * exp(-elA) + (1 - hv) * exp(-esA))) * exp(muB);
        return CB;
    }

    // =======================================================
    // source term phi
    KOKKOS_INLINE_FUNCTION
    real_t source_phi(Tag2Dilute tag, LBMState& lbmState) const
    {
        real_t dca = (exp(-elA) - exp(-esA));
        real_t dcb = (exp(-elB) - exp(-esB));
        real_t source_std = -lambda * Mphi / SQR(W) * p_prime(lbmState[IPHI]) * (elC - esC - exp(lbmState[IMU]) * dca - exp(lbmState[IMUB]) * dcb);
        real_t source_noct = -(1.0 - counter_term) * Mphi / SQR(W) * g_prime(lbmState[IPHI]);
        return (source_std + source_noct);
    }

    // =======================================================
    // advection terms for phase field
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        force_phi(Tag2Dilute tag, LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY]) + SQR(lbmState[IDPHIDZ])) + NORMAL_EPSILON;
        real_t force_ct = dx / dt * counter_term * 4.0 * lbmState[IPHI] * (1.0 - lbmState[IPHI]) / W / norm;
        RVect<dim> term;
        term[IX] = force_ct * lbmState[IDPHIDX];
        term[IY] = force_ct * lbmState[IDPHIDY];
        if (dim == 3) {
            term[IZ] = force_ct * lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of phase field equation
    KOKKOS_INLINE_FUNCTION
    real_t tau_phi(Tag2Dilute tag, LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 * Mphi * dt / SQR(dx));
        return (tau);
    }

    // =======================================================
    // source term muA
    KOKKOS_INLINE_FUNCTION
    real_t source_A(Tag2Dilute tag, LBMState& lbmState) const
    {
        const real_t mu = lbmState[IMU];
        real_t source_std = h_prime(lbmState[IPHI]) * (compute_CA(tag, 1.0, mu) - compute_CA(tag, 0.0, mu)) * lbmState[IDPHIDT];
        return (source_std);
    }
    // =======================================================
    // advection terms for diffusion equation
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        force_CA(Tag2Dilute tag, LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY]) + SQR(lbmState[IDPHIDZ])) + NORMAL_EPSILON;

        const real_t qv = q(lbmState[IPHI]);
        real_t force_lbm = diffusion_current * (DA1 - DA0) * (lbmState[IMU]) * lbmState[IC];
        real_t force_lbm2 = diffusion_current2 * FMAX(qv * DA1 + (1 - qv) * DA0, NORMAL_EPSILON) * lbmState[IMU];

        real_t force_at = at_current * W * 0.0 * lbmState[IDPHIDT] / norm;

        RVect<dim> term;
        term[IX] = dx / dt * gammaA * ((force_lbm + force_at) * lbmState[IDPHIDX] + force_lbm2 * lbmState[IDCADX]);
        term[IY] = dx / dt * gammaA * ((force_lbm + force_at) * lbmState[IDPHIDY] + force_lbm2 * lbmState[IDCADY]);
        if (dim == 3) {
            term[IZ] = dx / dt * gammaA * ((force_lbm + force_at) * lbmState[IDPHIDZ] + force_lbm2 * lbmState[IDCADZ]);
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t tau_CA(Tag2Dilute tag, LBMState& lbmState) const
    {
        //~ real_t tau = 0.5 + ( e2 * lbmState[IC] * dt / SQR(dx) / gammaA);
        real_t tau = 0.5 + (e2 * dt / SQR(dx) / gammaA);
        return (tau);
    }

    // =======================================================
    // susceptibility of mu diffusion
    KOKKOS_INLINE_FUNCTION
    real_t kiA(Tag2Dilute tag, LBMState& lbmState) const
    {
        //~ return (lbmState[IC]);
        real_t mu = lbmState[IMU];
        return (h(lbmState[IPHI]) * compute_CA(tag, 1.0, mu) + (1 - h(lbmState[IPHI])) * compute_CA(tag, 0.0, mu));
    }

    // =======================================================
    // mobility of the diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t compute_mobilityA(Tag2Dilute tag, LBMState& lbmState) const
    {
        const real_t qv = q(lbmState[IPHI]);
        //~ const real_t M = gammaA * (qv * DA1 + (1-qv) * DA0);
        const real_t M = gammaA * (qv * DA1 + (1 - qv) * DA0) * lbmState[IC];
        return M;
    }

    // =======================================================
    // source term muB
    KOKKOS_INLINE_FUNCTION
    real_t source_B(Tag2Dilute tag, LBMState& lbmState) const
    {
        const real_t mu = lbmState[IMUB];
        real_t source_std = h_prime(lbmState[IPHI]) * (compute_CB(tag, 1.0, mu) - compute_CB(tag, 0.0, mu)) * lbmState[IDPHIDT];
        return (source_std);
    }

    // =======================================================
    // advection terms for diffusion equation
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        force_CB(Tag2Dilute tag, LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY]) + SQR(lbmState[IDPHIDZ])) + NORMAL_EPSILON;
        const real_t qv = q(lbmState[IPHI]);
        real_t force_lbm = diffusion_current * (DB1 - DB0) * lbmState[IMUB] * lbmState[ICB];
        real_t force_lbm2 = diffusion_current2 * FMAX(qv * DB1 + (1 - qv) * DB0, NORMAL_EPSILON) * lbmState[IMUB];

        real_t force_at = at_current * W * 0.0 * lbmState[IDPHIDT] / norm;

        RVect<dim> term;
        term[IX] = dx / dt * gammaB * ((force_lbm + force_at) * lbmState[IDPHIDX] + force_lbm2 * lbmState[IDCBDX]);
        term[IY] = dx / dt * gammaB * ((force_lbm + force_at) * lbmState[IDPHIDY] + force_lbm2 * lbmState[IDCBDY]);
        if (dim == 3) {
            term[IZ] = dx / dt * gammaB * ((force_lbm + force_at) * lbmState[IDPHIDZ] + force_lbm2 * lbmState[IDCBDZ]);
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t tau_CB(Tag2Dilute tag, LBMState& lbmState) const
    {
        //~ real_t tau = 0.5 + ( e2 * lbmState[ICB]  * dt / SQR(dx) / gammaB);
        real_t tau = 0.5 + (e2 * dt / SQR(dx) / gammaB);
        return (tau);
    }

    // =======================================================
    // susceptibility of mu diffusion
    KOKKOS_INLINE_FUNCTION
    real_t kiB(Tag2Dilute tag, LBMState& lbmState) const
    {
        //~ return (lbmState[ICB]);
        real_t mu = lbmState[IMUB];
        return (h(lbmState[IPHI]) * compute_CB(tag, 1.0, mu) + (1 - h(lbmState[IPHI])) * compute_CB(tag, 0.0, mu));
    }

    // =======================================================
    // mobility of the diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t compute_mobilityB(Tag2Dilute tag, LBMState& lbmState) const
    {
        const real_t qv = q(lbmState[IPHI]);
        //~ const real_t M = gammaB * (qv * DB1 + (1-qv) * DB0);
        const real_t M = gammaB * (qv * DB1 + (1 - qv) * DB0) * lbmState[ICB];
        return M;
    }

}; // end struct model

}
#endif // MODELS_DISSOL_GP_MIXT_H_
