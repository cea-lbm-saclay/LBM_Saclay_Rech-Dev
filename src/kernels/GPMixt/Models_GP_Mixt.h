#ifndef MODELS_GP_MIXT_H_
#define MODELS_GP_MIXT_H_
#include "Models_GP_Mixt.h"
#include "GPMixt_Index.h"
#include <real_type.h>
#include "InitConditionsTypes.h"
namespace PBM_GP_MIXT {
// ================================================
//
// Models for mixt formulation of grand potential problems
//
// ================================================

struct ModelParams {

    using LBMState = typename Kokkos::Array<real_t, COMPONENT_SIZE>;
    //~ using localDataTypes;
    static constexpr real_t NORMAL_EPSILON = 1.0e-16;

    void showParams()
    {
        std::cout << "W :    " << W << std::endl;
        std::cout << "mailles par W :    " << W / dx << std::endl;
        std::cout << "Cseq :    " << Cseq << std::endl;
        std::cout << "Cleq :    " << Cleq << std::endl;
        std::cout << "mueq :    " << mueq << std::endl;
        std::cout << "counter_term :    " << counter_term << std::endl;
        std::cout << "at_current :    " << at_current << std::endl;
        std::cout << "init :" << initType << "." << std::endl;
        real_t tau = 0.5 + (e2 * Mphi * dt / SQR(dx));
        std::cout << "tauPhi :" << tau << std::endl;
        //~ std::cout << "tauPhiS :" << 0.5 + gammaTRT1 / (tau - 0.5) << std::endl;
        //~ real_t tauC = 0.5 + (e2 * initCl / gamma * dt / SQR(dx));
        std::cout << "tauCdilute_l :" << 0.5 + (e2 * initCl * D1 / gamma * dt / SQR(dx)) << std::endl;
        std::cout << "tauCdilute_s :" << 0.5 + (e2 * initCs * D0 / gamma * dt / SQR(dx)) << std::endl;
        //~ std::cout << "tauCS :" << 0.5 + gammaTRT2 / (tauC - 0.5) << std::endl;
        std::cout << "init dphidt :" << init_dphidt << std::endl;

    };
    ModelParams() {};
    ModelParams(const ConfigMap& configMap, LBMParams params)
    {

        dt = params.dt;
        dx = params.dx;
        e2 = configMap.getFloat("lbm", "e2", 3.0);
        gammaTRT1 = configMap.getFloat("equation1", "lambdaTRT", 1 / 12);
        gammaTRT2 = configMap.getFloat("equation2", "lambdaTRT", 1 / 12);
        W = configMap.getFloat("params", "W", 0.005);
        Mphi = configMap.getFloat("params", "Mphi", 1.2);
        lambda = configMap.getFloat("params", "lambda", 230.0);

        Cseq = configMap.getFloat("params", "Cseq", 0.6);
        Cleq = configMap.getFloat("params", "Cleq", 0.5);
        mueq = configMap.getFloat("params", "mueq", 0.4);
        D1 = configMap.getFloat("params", "D1", 1.0);
        D0 = configMap.getFloat("params", "D0", 0.0);
        gamma = configMap.getFloat("params", "gamma", 1.0);
        counter_term = configMap.getFloat("params", "counter_term", 0.0);
        at_current = configMap.getFloat("params", "at_current", 0.25);

        scale_chempot = configMap.getFloat("params", "scale_chempot", 0.0);

        el = configMap.getFloat("params", "el", 1.0);
        es = configMap.getFloat("params", "es", 1.0);
        epsinv = (el - es) / (el * es);

        Cs = configMap.getFloat("params", "Cs", 1.0);
        mu0s = configMap.getFloat("params", "mu0s", 0.0);
        cut_off = configMap.getFloat("params", "cut_off", 0.5);
        Q = configMap.getFloat("params", "Q", 0.04);

        surf = configMap.getFloat("params", "surf", 0.0);
        
        CC_phi_threshold = configMap.getFloat("params", "CC_phi_threshold", 1.0);
        
        

        x0 = configMap.getFloat("init", "x0", 0.0);
        y0 = configMap.getFloat("init", "y0", 0.0);
        z0 = configMap.getFloat("init", "z0", 0.0);
        r0 = configMap.getFloat("init", "r0", 0.2);

        x2 = configMap.getFloat("init", "x2", 0.0);
        y2 = configMap.getFloat("init", "y2", 0.0);
        z2 = configMap.getFloat("init", "z2", 0.0);
        r2 = configMap.getFloat("init", "r2", 0.2);

        initCl = configMap.getFloat("init", "initCl", 0.1);
        initCs = configMap.getFloat("init", "initCs", 0.2);
        sign = configMap.getFloat("init", "sign", 1);
        t0 = configMap.getFloat("init", "t0", 1.0);
        initErfc = configMap.getFloat("init", "initErfc", 0.1);
        xi = configMap.getFloat("init", "xi", 0.0);
        phi0_fact = configMap.getFloat("init", "phi0", 1.0);
        init_dphidt = configMap.getFloat("init", "dphidt", 0.0);
        
        

        use_sharp_init = configMap.getBool("init", "sharp_init", false);

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

        showParams();
    }

    //! model params

    real_t dx, dt, e2, time; // simulation params
    real_t gammaTRT1, gammaTRT2; // trt params
    real_t W, Mphi, lambda, counter_term; // general params for phi equation
    real_t D1, D0, gamma, at_current, scale_chempot; // general params for C equation
    real_t Cseq, Cleq, mueq; // params for quadratic free energies
    real_t el, es, epsinv, muPlus, muMinus; // params for quadratic sloped free energies
    real_t Cs, mu0s, cut_off, Q; // params for bringedal model
    real_t surf; // params for bringedal model
    
    real_t CC_phi_threshold; // params for infinite diff
    //! init params
    real_t x0, y0, z0, r0, x2, y2, z2, r2, initCl, initCs, sign, t0, initErfc, xi, phi0_fact, init_dphidt;
    int initType;
    bool use_sharp_init;

    KOKKOS_INLINE_FUNCTION real_t phi0(real_t x) const { return phi0_fact * 0.5 * (1 + tanh(sign * 2.0 * x / W)); }

    // interpolation functions

    KOKKOS_INLINE_FUNCTION real_t p(real_t phi) const { return SQR(phi) * (3.0 - 2.0 * phi); }
    KOKKOS_INLINE_FUNCTION real_t p_prime(real_t phi) const { return 6.0 * phi * (1.0 - phi); }
    KOKKOS_INLINE_FUNCTION real_t fun_g(real_t phi) const { return 8 * SQR(phi) * SQR(1.0 - phi); }
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
    // compute mu
    KOKKOS_INLINE_FUNCTION
    real_t compute_mu(Tag2Quadra tag, real_t phi, real_t C) const
    {
        const real_t hv = h(phi);
        const real_t mu = mueq + C - Cleq * hv - Cseq * (1.0 - hv) - surf * sqrt(2 * fun_g(phi));
        return mu;
    }
    KOKKOS_INLINE_FUNCTION
    real_t compute_cl(Tag2Quadra tag, real_t mu) const
    {
        const real_t cl = (mu+Cleq-mueq);
        return cl;
    }
    KOKKOS_INLINE_FUNCTION
    real_t compute_cs(Tag2Quadra tag, real_t mu) const
    {
		const real_t cs = (mu+Cseq-mueq);
        return cs;
    }

    // =======================================================
    // compute chemical grand potential
    KOKKOS_INLINE_FUNCTION
    real_t compute_w(Tag2Quadra tag, const LBMState& lbmState) const
    {
        const real_t mu = lbmState[IMU];
        const real_t phi = lbmState[IPHI];
        //~ const real_t C = lbmState[IC];
        
        const real_t ml=Cleq+mueq;
        const real_t ms=Cseq+mueq;
        const real_t dQ=mueq * (Cleq-Cseq);

        const real_t w = -0.5 * SQR(mu) - mu * (phi * ml + (1-phi) * ms) + phi * dQ; // défini à constante près

        return w;
    }
    // =======================================================
    // compute grand potential energy
    KOKKOS_INLINE_FUNCTION
    real_t compute_gp_energy(Tag2Quadra tag, const LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY]) + SQR(lbmState[IDPHIDZ])) + NORMAL_EPSILON;
        const real_t phi = lbmState[IPHI];
        const real_t w = compute_w(tag, lbmState);

        const real_t gp = 0.5 * SQR(W) / lambda * SQR(norm) + fun_g(phi) / lambda + w;

        return gp;
    }
    // =======================================================
    // compute helmoltz free energy
    KOKKOS_INLINE_FUNCTION
    real_t compute_helmoltz_energy(Tag2Quadra tag, const LBMState& lbmState) const
    {
        const real_t mu = lbmState[IMU];
        const real_t C = lbmState[IC];

        const real_t w = compute_w(tag, lbmState);
        const real_t f = w + mu * C;

        return f;
    }

    // =======================================================
    // =======================================================
    //
    //		 equation 1 : phase field
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2Quadra type, EquationTag1 tag, const LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2Quadra type, EquationTag1 tag, const LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }
    // =======================================================
    // source term phi
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2Quadra type, EquationTag1 tag, const LBMState& lbmState) const
    {
        real_t source_std = lambda * Mphi / SQR(W) * p_prime(lbmState[IPHI]) * (Cleq - Cseq) * (lbmState[IMU] - mueq);
        real_t source_noct = -(1.0 - counter_term) * Mphi / SQR(W) * g_prime(lbmState[IPHI]);
        return (source_std + source_noct);
    }

    // =======================================================
    // advection terms for phase field
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        S1(Tag2Quadra type, EquationTag1 tag, const LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY])) + NORMAL_EPSILON;
        real_t force_ct = Mphi * counter_term * 4.0 * lbmState[IPHI] * (1.0 - lbmState[IPHI]) / W / norm;
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
    real_t tau(Tag2Quadra type, EquationTag1 tag, const LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 * Mphi * dt / SQR(dx));
        return (tau);
    }

    // =======================================================
    // =======================================================
    //
    //		 equation 2 : composition
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2Quadra type, EquationTag2 tag, const LBMState& lbmState) const
    {
        return lbmState[IC];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2Quadra type, EquationTag2 tag, const LBMState& lbmState) const
    {
        const real_t qv = q(lbmState[IPHI]);
        const real_t M = gamma * (qv * D1 + (1 - qv) * D0) / D1 * (lbmState[IMU] - scale_chempot * mueq);
        return M;
    }

    // =======================================================
    // source term C
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2Quadra type, EquationTag2 tag, const LBMState& lbmState) const
    {
        return 0.0;
    }
    // =======================================================
    // advection terms for diffusion equation
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        S1(Tag2Quadra type, EquationTag2 tag, const LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY])) + NORMAL_EPSILON;
        real_t force_lbm = (D1 - D0) * (lbmState[IMU] - scale_chempot * mueq);
        real_t force_at = at_current * W * (Cseq - Cleq) * lbmState[IDPHIDT] / norm;

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
    real_t tau(Tag2Quadra type, EquationTag2 tag, const LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 / gamma * dt / SQR(dx));
        return (tau);
    }

    // ================================================
    //
    // functions for the case of 2 quadratic free energies
    //
    // ================================================
    struct Tag2QuadraSloped {
    };

    // =======================================================
    // compute mu
    KOKKOS_INLINE_FUNCTION
    real_t compute_mu(Tag2QuadraSloped tag, real_t phi, real_t C) const
    {
        const real_t hv = h(phi);
        const real_t chi = 1 / es - phi * epsinv;
        const real_t mu = mueq + (C - Cleq * hv - Cseq * (1.0 - hv)) / chi;
        return mu;
    }
    
    KOKKOS_INLINE_FUNCTION
    real_t compute_cl(Tag2QuadraSloped tag, real_t mu) const
    {
        const real_t cl = (mu+Cleq-mueq);
        return cl;
    }
    KOKKOS_INLINE_FUNCTION
    real_t compute_cs(Tag2QuadraSloped tag, real_t mu) const
    {
		const real_t cs = (mu+Cseq-mueq);
        return cs;
    }

    // =======================================================
    // compute chemical grand potential
    KOKKOS_INLINE_FUNCTION
    real_t compute_w(Tag2QuadraSloped tag, const LBMState& lbmState) const
    {
        const real_t mu = lbmState[IMU];
        const real_t phi = lbmState[IPHI];
        //~ const real_t C = lbmState[IC];

        const real_t ml = Cleq - mueq / el;
        const real_t ms = Cseq - mueq / es;
        const real_t Ql = mueq * (0.5 * epsinv * mueq - (Cleq - Cseq));

        const real_t wl = -0.5 * SQR(mu) / el - mu * ml + Ql;
        const real_t ws = -0.5 * SQR(mu) / es - mu * ms;

        const real_t w = p(phi) * wl + (1 - p(phi)) * ws;

        return w;
    }
    // =======================================================
    // compute grand potential energy
    KOKKOS_INLINE_FUNCTION
    real_t compute_gp_energy(Tag2QuadraSloped tag, const LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY]) + SQR(lbmState[IDPHIDZ])) + NORMAL_EPSILON;
        const real_t phi = lbmState[IPHI];
        const real_t w = compute_w(tag, lbmState);

        const real_t gp = 0.5 * SQR(W) / lambda * SQR(norm) + fun_g(phi) / lambda + w;

        return gp;
    }
    // =======================================================
    // compute helmoltz free energy
    KOKKOS_INLINE_FUNCTION
    real_t compute_helmoltz_energy(Tag2QuadraSloped tag, const LBMState& lbmState) const
    {
        const real_t mu = lbmState[IMU];
        const real_t C = lbmState[IC];

        const real_t w = compute_w(tag, lbmState);
        const real_t f = w + mu * C;

        return f;
    }

    // =======================================================
    // =======================================================
    //
    //		 equation 1 : phase field
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2QuadraSloped type, EquationTag1 tag, const LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2QuadraSloped type, EquationTag1 tag, const LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }
    // =======================================================
    // source term phi
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2QuadraSloped type, EquationTag1 tag, const LBMState& lbmState) const
    {

        const real_t deltaw = (lbmState[IMU] - mueq) * (0.5 * epsinv * (lbmState[IMU] - mueq) - (Cleq - Cseq));
        const real_t source_std = lambda * Mphi / SQR(W) * p_prime(lbmState[IPHI]) * deltaw;
        const real_t source_noct = -(1.0 - counter_term) * Mphi / SQR(W) * g_prime(lbmState[IPHI]);
        return (source_std + source_noct);
    }

    // =======================================================
    // advection terms for phase field
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        S1(Tag2QuadraSloped type, EquationTag1 tag, const LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY])) + NORMAL_EPSILON;
        real_t force_ct = Mphi * counter_term * 4.0 * lbmState[IPHI] * (1.0 - lbmState[IPHI]) / W / norm;
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
    real_t tau(Tag2QuadraSloped type, EquationTag1 tag, const LBMState& lbmState) const
    {

        const real_t tau = 0.5 + (e2 * Mphi * dt / SQR(dx));
        return (tau);
    }

    // =======================================================
    // =======================================================
    //
    //		 equation 2 : composition
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2QuadraSloped type, EquationTag2 tag, const LBMState& lbmState) const
    {
        return lbmState[IC];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2QuadraSloped type, EquationTag2 tag, const LBMState& lbmState) const
    {
        const real_t qv = q(lbmState[IPHI]);
        const real_t M = gamma * (qv * D1 + (1 - qv) * D0) / D1 * (lbmState[IMU] - scale_chempot * mueq);
        return M;
    }

    // =======================================================
    // source term C
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2QuadraSloped type, EquationTag2 tag, const LBMState& lbmState) const
    {
        return 0.0;
    }
    // =======================================================
    // advection terms for diffusion equation
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        S1(Tag2QuadraSloped type, EquationTag2 tag, const LBMState& lbmState) const
    {
        const real_t phi = lbmState[IPHI];
        const real_t chi = 1 / es - phi * epsinv;
        const real_t D = D1 * phi + (1 - phi) * D0;
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY])) + NORMAL_EPSILON;

        //~ const real_t force_lbm = (D1 - D0) * (lbmState[IMU] - scale_chempot*mueq);
        const real_t force_lbm = (lbmState[IMU] - scale_chempot * mueq) * ((D1 - D0) * chi + epsinv * D);
        const real_t force_at = at_current * W * (Cseq - Cleq) * lbmState[IDPHIDT] / norm;

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
    real_t tau(Tag2QuadraSloped type, EquationTag2 tag, const LBMState& lbmState) const
    {
        //~ const real_t phi = lbmState[IPHI];
        //~ const real_t chi = 1/es-phi*epsinv;
        const real_t tau = 0.5 + (e2 / gamma * dt / SQR(dx));
        return (tau);
    }

    // ================================================
    //
    // functions for the case of 2 dilute free energies
    //
    // ================================================
    struct Tag2Dilute {
    };

    // =======================================================
    // compute mu
    KOKKOS_INLINE_FUNCTION
    real_t compute_mu(Tag2Dilute tag, real_t phi, real_t C) const
    {
        const real_t hv = h(phi);
        real_t mu;
        if (C > 0) {
            mu = mueq + log(C / (hv * Cleq + (1 - hv) * Cseq));
        } else {
            mu = -100;
        }
        return mu;
    }
    
    KOKKOS_INLINE_FUNCTION
    real_t compute_cl(Tag2Dilute tag, real_t mu) const
    {
        const real_t cl = (Cleq)*exp(mu-mueq);
        return cl;
    }
    KOKKOS_INLINE_FUNCTION
    real_t compute_cs(Tag2Dilute tag, real_t mu) const
    {
		const real_t cs = (Cseq)*exp(mu-mueq);
        return cs;
    }

    // =======================================================
    // compute chemical grand potential
    KOKKOS_INLINE_FUNCTION
    real_t compute_w(Tag2Dilute tag, const LBMState& lbmState) const
    {
        const real_t mu = lbmState[IMU];
        const real_t phi = lbmState[IPHI];
        //~ const real_t C = lbmState[IC];

        const real_t w  =  phi * (Cleq - Cseq) - exp(mu-mueq)* (phi*Cleq+(1-phi)*Cseq); // lacks a constant that cannot be determined from mueq, cleq and cseq)

        return w;
    }
    // =======================================================
    // compute grand potential energy
    KOKKOS_INLINE_FUNCTION
    real_t compute_gp_energy(Tag2Dilute tag, const LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY]) + SQR(lbmState[IDPHIDZ])) + NORMAL_EPSILON;
        const real_t phi = lbmState[IPHI];
        const real_t w = compute_w(tag, lbmState);

        const real_t gp = 0.5 * SQR(W) / lambda * SQR(norm) + fun_g(phi) / lambda + w;

        return gp;
    }
    // =======================================================
    // compute helmoltz free energy
    KOKKOS_INLINE_FUNCTION
    real_t compute_helmoltz_energy(Tag2Dilute tag, const LBMState& lbmState) const
    {
        const real_t mu = lbmState[IMU];
        const real_t C = lbmState[IC];

        const real_t w = compute_w(tag, lbmState);
        const real_t f = w + mu * C;

        return f;
    }

    // =======================================================
    // =======================================================
    //
    //		 equation 1 : phase field
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
        real_t source_std = lambda * Mphi / SQR(W) * p_prime(lbmState[IPHI]) * (Cleq - Cseq) * (exp(lbmState[IMU] - mueq) - 1);
        real_t source_noct = -(1.0 - counter_term) * Mphi / SQR(W) * g_prime(lbmState[IPHI]);
        return (source_std + source_noct);
    }

    // =======================================================
    // advection terms for phase field
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        S1(Tag2Dilute type, EquationTag1 tag, const LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY])) + NORMAL_EPSILON;
        real_t force_ct = Mphi * counter_term * 4.0 * lbmState[IPHI] * (1.0 - lbmState[IPHI]) / W / norm;
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
    real_t tau(Tag2Dilute type, EquationTag1 tag, const LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 * Mphi * dt / SQR(dx));
        return (tau);
    }

    // =======================================================
    // =======================================================
    //
    //		 equation 2 : composition
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2Dilute type, EquationTag2 tag, LBMState& lbmState) const
    {
        return lbmState[IC];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2Dilute type, EquationTag2 tag, LBMState& lbmState) const
    {
        
        const real_t M = gamma *  (lbmState[IMU] - scale_chempot * mueq);
        return M;
    }

    // =======================================================
    // source term phi
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2Dilute type, EquationTag2 tag, LBMState& lbmState) const
    {
        return 0.0;
    }
    // =======================================================
    // advection terms for diffusion equation
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        S1(Tag2Dilute type, EquationTag2 tag, LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY])) + NORMAL_EPSILON;
        real_t force_lbm = 0;
        real_t force_at = at_current * W * (Cseq - Cleq) * lbmState[IDPHIDT] / norm;
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
    real_t tau(Tag2Dilute type, EquationTag2 tag, LBMState& lbmState) const
    {
		const real_t qv = q(lbmState[IPHI]);
        real_t tau = 0.5 + (lbmState[IC] * (qv * D1 + (1 - qv) * D0) * e2 / gamma * dt / SQR(dx));
        return (tau);
    }
    
    
    // ================================================
    //
    // functions for the case of 2 dilute free energies
    //
    // ================================================
    struct Tag2DiluteNoSolidDiff {
    };

    // =======================================================
    // compute mu
    KOKKOS_INLINE_FUNCTION
    real_t compute_mu(Tag2DiluteNoSolidDiff tag, real_t phi, real_t C) const
    {
        const real_t hv = h(phi);
        real_t mu;
        if (C > 0) {
            mu = mueq + log(C / (hv * Cleq + (1 - hv) * Cseq));
        } else {
            mu = -100;
        }
        return mu;
    }
    KOKKOS_INLINE_FUNCTION
    real_t compute_cl(Tag2DiluteNoSolidDiff tag, real_t mu) const
    {
        const real_t cl = (Cleq)*exp(mu-mueq);
        return cl;
    }
    KOKKOS_INLINE_FUNCTION
    real_t compute_cs(Tag2DiluteNoSolidDiff tag, real_t mu) const
    {
		const real_t cs = (Cseq)*exp(mu-mueq);
        return cs;
    }
    // =======================================================
    // compute chemical grand potential
    KOKKOS_INLINE_FUNCTION
    real_t compute_w(Tag2DiluteNoSolidDiff tag, const LBMState& lbmState) const
    {
        const real_t mu = lbmState[IMU];
        const real_t phi = lbmState[IPHI];
        //~ const real_t C = lbmState[IC];

        const real_t w  =  phi * (Cleq - Cseq) - exp(mu-mueq)* (phi*Cleq+(1-phi)*Cseq); // lacks a constant that cannot be determined from mueq, cleq and cseq)

        return w;
    }
    // =======================================================
    // compute grand potential energy
    KOKKOS_INLINE_FUNCTION
    real_t compute_gp_energy(Tag2DiluteNoSolidDiff tag, const LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY]) + SQR(lbmState[IDPHIDZ])) + NORMAL_EPSILON;
        const real_t phi = lbmState[IPHI];
        const real_t w = compute_w(tag, lbmState);

        const real_t gp = 0.5 * SQR(W) / lambda * SQR(norm) + fun_g(phi) / lambda + w;

        return gp;
    }
    // =======================================================
    // compute helmoltz free energy
    KOKKOS_INLINE_FUNCTION
    real_t compute_helmoltz_energy(Tag2DiluteNoSolidDiff tag, const LBMState& lbmState) const
    {
        const real_t mu = lbmState[IMU];
        const real_t C = lbmState[IC];

        const real_t w = compute_w(tag, lbmState);
        const real_t f = w + mu * C;

        return f;
    }

    // =======================================================
    // =======================================================
    //
    //		 equation 1 : phase field
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2DiluteNoSolidDiff type, EquationTag1 tag, const LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2DiluteNoSolidDiff type, EquationTag1 tag, const LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }

    // =======================================================
    // source term phi
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2DiluteNoSolidDiff type, EquationTag1 tag, const LBMState& lbmState) const
    {
        real_t source_std = lambda * Mphi / SQR(W) * p_prime(lbmState[IPHI]) * (Cleq - Cseq) * (exp(lbmState[IMU] - mueq) - 1);
        real_t source_noct = -(1.0 - counter_term) * Mphi / SQR(W) * g_prime(lbmState[IPHI]);
        return (source_std + source_noct);
    }

    // =======================================================
    // advection terms for phase field
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        S1(Tag2DiluteNoSolidDiff type, EquationTag1 tag, const LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY])) + NORMAL_EPSILON;
        real_t force_ct = Mphi * counter_term * 4.0 * lbmState[IPHI] * (1.0 - lbmState[IPHI]) / W / norm;
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
    real_t tau(Tag2DiluteNoSolidDiff type, EquationTag1 tag, const LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 * Mphi * dt / SQR(dx));
        return (tau);
    }

    // =======================================================
    // =======================================================
    //
    //		 equation 2 : composition
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2DiluteNoSolidDiff type, EquationTag2 tag, LBMState& lbmState) const
    {
        return lbmState[IC];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2DiluteNoSolidDiff type, EquationTag2 tag, LBMState& lbmState) const
    {
        const real_t qv = q(lbmState[IPHI]);
        const real_t M = gamma * (qv * D1 + (1 - qv) * D0) * (lbmState[IMU] - scale_chempot * mueq);
        return M;
    }

    // =======================================================
    // source term phi
    KOKKOS_INLINE_FUNCTION
    real_t S0(Tag2DiluteNoSolidDiff type, EquationTag2 tag, LBMState& lbmState) const
    {
        return 0.0;
    }
    // =======================================================
    // advection terms for diffusion equation
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        S1(Tag2DiluteNoSolidDiff type, EquationTag2 tag, LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY])) + NORMAL_EPSILON;
        real_t force_lbm = (D1 - D0) * lbmState[IC] * (lbmState[IMU] - scale_chempot * mueq);
        real_t force_at = at_current * W * (Cseq - Cleq) * lbmState[IDPHIDT] / norm;
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
    real_t tau(Tag2DiluteNoSolidDiff type, EquationTag2 tag, LBMState& lbmState) const
    {
        real_t tau = 0.5 + (lbmState[IC] * e2 / gamma * dt / SQR(dx));
        return (tau);
    }

    // ================================================
    //
    // functions for the case of 1 quadratic and 1 linear free energies (Bringedal equivalent)
    //
    // ================================================
    struct TagGPBringedal {
    };

    // =======================================================
    // compute chemical potential
    KOKKOS_INLINE_FUNCTION
    real_t compute_mu(TagGPBringedal tag, real_t phi, real_t C) const
    {
        real_t mu;
        if (phi < cut_off)
            mu = mueq;
        else
            mu = mueq - Cleq + C / (phi + NORMAL_EPSILON);
        return mu;
    }
    KOKKOS_INLINE_FUNCTION
    real_t compute_cl(TagGPBringedal tag, real_t mu) const
    {
        const real_t cl = (mu+Cleq-mueq);
        return cl;
    }
    KOKKOS_INLINE_FUNCTION
    real_t compute_cs(TagGPBringedal tag, real_t mu) const
    {
		const real_t cs = (mu+Cseq-mueq);
        return cs;
    }
    // =======================================================
    // compute chemical grand potential
    KOKKOS_INLINE_FUNCTION
    real_t compute_w(TagGPBringedal tag, LBMState& lbmState) const
    {
        const real_t mu = lbmState[IMU];
        const real_t phi = lbmState[IPHI];
        const real_t C = lbmState[IC];

        const real_t w = -0.5 * SQR(mu) - mu * (C - mu) + phi * (Cleq - Cseq) * mueq;

        return w;
    }
    // =======================================================
    // compute grand potential energy
    KOKKOS_INLINE_FUNCTION
    real_t compute_gp_energy(TagGPBringedal tag, LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY]) + SQR(lbmState[IDPHIDZ])) + NORMAL_EPSILON;
        const real_t phi = lbmState[IPHI];
        const real_t w = compute_w(tag, lbmState);

        const real_t gp = 0.5 * SQR(norm) + fun_g(phi) / lambda + w;

        return gp;
    }
    // =======================================================
    // compute helmoltz free energy
    KOKKOS_INLINE_FUNCTION
    real_t compute_helmoltz_energy(TagGPBringedal tag, LBMState& lbmState) const
    {
        const real_t mu = lbmState[IMU];
        const real_t C = lbmState[IC];

        const real_t w = compute_w(tag, lbmState);
        const real_t f = w + mu * C;

        return f;
    }

    // =======================================================
    // =======================================================
    //
    //		 equation 1 : phase field
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(TagGPBringedal type, EquationTag1 tag, LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(TagGPBringedal type, EquationTag1 tag, LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }

    // =======================================================
    // source term phi
    KOKKOS_INLINE_FUNCTION
    real_t S0(TagGPBringedal type, EquationTag1 tag, LBMState& lbmState) const
    {
        real_t source_std = lambda * Mphi / SQR(W) * p_prime(lbmState[IPHI]) * 0.5 * (SQR(Cleq + lbmState[IMU] - mueq) - SQR(Cleq));
        real_t source_noct = -(1.0 - counter_term) * Mphi / SQR(W) * g_prime(lbmState[IPHI]);
        return (source_std + source_noct);
    }

    // =======================================================
    // advection terms for phase field
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        S1(TagGPBringedal type, EquationTag1 tag, LBMState& lbmState) const
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
    real_t tau(TagGPBringedal type, EquationTag1 tag, LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 * Mphi * dt / SQR(dx));
        return (tau);
    }

    // =======================================================
    // =======================================================
    //
    //		 equation 2 : composition
    //
    // =======================================================

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(TagGPBringedal type, EquationTag2 tag, LBMState& lbmState) const
    {
        return lbmState[IC];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(TagGPBringedal type, EquationTag2 tag, LBMState& lbmState) const
    {
        const real_t qv = q(lbmState[IPHI]);
        const real_t M = gamma * (qv * D1 + (1 - qv) * D0) * (lbmState[IMU] - scale_chempot * mueq);
        return M;
    }

    // =======================================================
    // source term phi
    KOKKOS_INLINE_FUNCTION
    real_t S0(TagGPBringedal type, EquationTag2 tag, LBMState& lbmState) const
    {
        return 0.0;
    }
    // =======================================================
    // advection terms for diffusion equation
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        S1(TagGPBringedal type, EquationTag2 tag, LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY]) + SQR(lbmState[IDPHIDZ])) + NORMAL_EPSILON;
        real_t force_lbm = D1 * lbmState[IMU];
        real_t force_at = at_current * W * lbmState[IDPHIDT] / norm;
        RVect<dim> term;
        term[IX] = dx / dt * gamma * (force_lbm + force_at) * lbmState[IDPHIDX];
        term[IY] = dx / dt * gamma * (force_lbm + force_at) * lbmState[IDPHIDY];
        if (dim == 3) {
            term[IZ] = dx / dt * gamma * (force_lbm + force_at) * lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t tau(TagGPBringedal type, EquationTag2 tag, LBMState& lbmState) const
    {
        real_t tau = 0.5 + (lbmState[IC] * e2 / gamma * dt / SQR(dx));
        return (tau);
    }

}; // end struct model

} //end namespace
#endif // MODELS_DISSOL_GP_MIXT_H_
