#ifndef MODELS_GP_MIXT_NS_H_
#define MODELS_GP_MIXT_NS_H_

#include "GPMixt_NS_Index.h"
#include <real_type.h>
namespace PBM_GP_MIXT_NS {
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
        std::cout << "Cseq :    " << Cseq << std::endl;
        std::cout << "Cleq :    " << Cleq << std::endl;
        std::cout << "mueq :    " << mueq << std::endl;
        std::cout << "tauphi :    " << 0.5 + (e2 * Mphi * dt / SQR(dx)) << std::endl;
        std::cout << "tauNS :    " << (0.5 + (e2 * nuL * dt / SQR(dx))) << std::endl;
        std::cout << "tauC :    " << (0.5 + (e2 / gamma * dt / SQR(dx))) << std::endl;
        std::cout << "cs :    " << ((dx / dt) / SQRT(e2)) << std::endl;
        std::cout << "cs2 :    " << SQR(dx / dt) / e2 << std::endl;
        std::cout << "counter_term :    " << counter_term << std::endl;
        std::cout << "init :" << initType << "." << std::endl;
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
        Q = configMap.getFloat("params", "Q", 0.04);
        Cseq = configMap.getFloat("params", "Cseq", 0.6);
        Cleq = configMap.getFloat("params", "Cleq", 0.5);
        mueq = configMap.getFloat("params", "mueq", 0.4);
        D1 = configMap.getFloat("params", "D1", 1.0);
        D0 = configMap.getFloat("params", "D0", 0.0);
        gamma = configMap.getFloat("params", "gamma", 1.0);
        counter_term = configMap.getFloat("params", "counter_term", 0.0);
        at_current = configMap.getFloat("params", "at_current", 0.25);

        Cleq = configMap.getFloat("params", "Cleq", 0.5);
        Cs = configMap.getFloat("params", "Cs", 1.0);
        mu0s = configMap.getFloat("params", "mu0s", 0.0);
        cut_off = configMap.getFloat("params", "cut_off", 0.5);
        scale_chempot = configMap.getFloat("params", "scale_chempot", 0.0);

        rhoS = configMap.getFloat("params", "rhoS", 1.0);
        rhoL = configMap.getFloat("params", "rhoL", 1.0);

        nuS = configMap.getFloat("params", "nuS", 1.0);
        nuL = configMap.getFloat("params", "nuL", 1.0);

        fx = configMap.getFloat("params", "fx", 0.0);
        fy = configMap.getFloat("params", "fy", 0.0);
        fz = configMap.getFloat("params", "fz", 0.0);
        gx = configMap.getFloat("params", "gx", 0.0);
        gy = configMap.getFloat("params", "gy", 0.0);
        gz = configMap.getFloat("params", "gz", 0.0);
        hx = configMap.getFloat("params", "hx", 0.0);
        hy = configMap.getFloat("params", "hy", 0.0);
        hz = configMap.getFloat("params", "hz", 0.0);
        sigma = configMap.getFloat("params", "sigma", 0.0);

        PF_advec = configMap.getFloat("params", "PF_advec", 0.0);
        COMP_advec = configMap.getFloat("params", "COMP_advec", 0.0);
        shape_coef = configMap.getFloat("params", "shape_coef", 0.0);
        hstar = configMap.getFloat("params", "hstar", 0.0);

        x0 = configMap.getFloat("init", "x0", 0.0);
        y0 = configMap.getFloat("init", "y0", 0.0);
        z0 = configMap.getFloat("init", "z0", 0.0);
        r0 = configMap.getFloat("init", "r0", 0.2);
        initCl = configMap.getFloat("init", "initCl", 0.1);
        initCs = configMap.getFloat("init", "initCs", 0.2);
        sign = configMap.getFloat("init", "sign", 1);
        t0 = configMap.getFloat("init", "t0", 1.0);
        initErfc = configMap.getFloat("init", "initErfc", 0.1);
        xi = configMap.getFloat("init", "xi", 0.0);

        initVX = configMap.getFloat("init", "initVX", 0.2);
        initVY = configMap.getFloat("init", "initVY", 0.2);
        initVZ = configMap.getFloat("init", "initVZ", 0.2);

        initType = PHASE_FIELD_INIT_UNDEFINED;
        std::string initTypeStr = std::string(configMap.getString("init", "init_type", "unknown"));

        if (initTypeStr == "vertical")
            initType = PHASE_FIELD_INIT_VERTICAL;
        else if (initTypeStr == "sphere")
            initType = PHASE_FIELD_INIT_SPHERE;
        else if (initTypeStr == "square")
            initType = PHASE_FIELD_INIT_SQUARE;
        else if (initTypeStr == "data")
            initType = PHASE_FIELD_INIT_DATA;

        showParams();
    }

    //! model params
    real_t W, Mphi, lambda, Q, D1, D0, gamma, counter_term, at_current, Cs, mu0s, cut_off;
    real_t Cseq, Cleq, mueq, scale_chempot;
    real_t dx, dt, e2, cs2;
    real_t nuL, rhoL, nuS, rhoS, shape_coef, hstar;
    real_t fx, fy, fz, gx, gy, gz, hx, hy, hz, sigma;
    real_t PF_advec, COMP_advec;
    //! init params
    real_t x0, y0, z0, r0, initCl, initCs, sign, t0, initErfc, xi;
    real_t initVX, initVY, initVZ;
    int initType;

    KOKKOS_INLINE_FUNCTION real_t phi0(real_t x) const { return 0.5 * (1 + tanh(sign * 2.0 * x / W)); }
    KOKKOS_INLINE_FUNCTION real_t interp_rho(real_t phi) const
    {
        return (1.0 - phi) * rhoS + (phi * rhoL);
        //~ return rho0*rho1/( ((1.0-phi) * rho1) + ((phi) * rho0));
    }
    // interpolation functions
    KOKKOS_INLINE_FUNCTION real_t p(real_t phi) const { return SQR(phi) * (3.0 - 2.0 * phi); }
    KOKKOS_INLINE_FUNCTION real_t p_prime(real_t phi) const { return 6.0 * phi * (1.0 - phi); }
    KOKKOS_INLINE_FUNCTION real_t g_prime(real_t phi) const { return 16.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi); }
    KOKKOS_INLINE_FUNCTION real_t h(real_t phi) const { return phi; }
    KOKKOS_INLINE_FUNCTION real_t h_prime(real_t phi) const { return 1.0; }
    KOKKOS_INLINE_FUNCTION real_t q(real_t phi) const { return phi; }
    KOKKOS_INLINE_FUNCTION real_t q_prime(real_t phi) const { return 1.0; }
    KOKKOS_INLINE_FUNCTION real_t aa(real_t phi) const { return 1.0; }

    // ================================================
    //
    // functions for the case of 2 quadratic free energies
    //
    // ================================================
    struct Tag2Quadra {
    };

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
        real_t source_std = lambda * Mphi / SQR(W) * p_prime(lbmState[IPHI]) * (Cleq - Cseq) * (lbmState[IMU] - mueq);
        real_t source_noct = -(1.0 - counter_term) * Mphi / SQR(W) * g_prime(lbmState[IPHI]);
        return (source_std + source_noct);
    }

    // =======================================================
    // advection terms for phase field
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        S1(Tag2Quadra type, EquationTag1 tag, LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY])) + NORMAL_EPSILON;
        real_t force_ct = Mphi * counter_term * 4.0 * lbmState[IPHI] * (1.0 - lbmState[IPHI]) / W / norm;
        RVect<dim> term;
        term[IX] = force_ct * lbmState[IDPHIDX] + PF_advec * lbmState[IPHI] * lbmState[IU];
        term[IY] = force_ct * lbmState[IDPHIDY] + PF_advec * lbmState[IPHI] * lbmState[IV];
        if (dim == 3) {
            term[IZ] = force_ct * lbmState[IDPHIDZ] + PF_advec * lbmState[IPHI] * lbmState[IW];
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
    // compute mu
    KOKKOS_INLINE_FUNCTION
    real_t compute_mu(Tag2Quadra tag, real_t phi, real_t C) const
    {
        const real_t hv = h(phi);
        const real_t mu = mueq + C - Cleq * hv - Cseq * (1.0 - hv);
        return mu;
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
    real_t M0(Tag2Quadra type, EquationTag2 tag, LBMState& lbmState) const
    {
        return lbmState[IC];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2Quadra type, EquationTag2 tag, LBMState& lbmState) const
    {
        const real_t qv = q(lbmState[IPHI]);
        const real_t M = gamma * (qv * D1 + (1 - qv) * D0) / D1 * (lbmState[IMU] - scale_chempot * mueq);
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
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        S1(Tag2Quadra type, EquationTag2 tag, LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY])) + NORMAL_EPSILON;
        const real_t force_lbm = (D1 - D0) * (lbmState[IMU] - scale_chempot * mueq);
        const real_t force_at = at_current * W * (Cseq - Cleq) * lbmState[IDPHIDT] / norm;
        const real_t force_advec = COMP_advec * lbmState[IC] * lbmState[IPHI];
        RVect<dim> term;
        term[IX] = (force_lbm + force_at) * lbmState[IDPHIDX] + force_advec * lbmState[IU];
        term[IY] = (force_lbm + force_at) * lbmState[IDPHIDY] + force_advec * lbmState[IV];
        if (dim == 3) {
            term[IZ] = (force_lbm + force_at) * lbmState[IDPHIDZ] + force_advec * lbmState[IW];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t tau(Tag2Quadra type, EquationTag2 tag, LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 / gamma * dt / SQR(dx));
        return (tau);
    }

    // =======================================================
    // =======================================================
    //
    //		 equation 2 : navier stokes
    //
    // =======================================================

    // =======================================================
    // force term for ns equation
    KOKKOS_INLINE_FUNCTION
    RVect2 force_NS(Tag2Quadra tag, LBMState& lbmState) const
    {
        const real_t rho = lbmState[ID];
        const real_t phi = lbmState[IPHI];
        const real_t tension = sigma * 1.5 / W * (g_prime(phi) - SQR(W) * lbmState[ILAPLAPHI]);
        RVect2 term;
        term[IX] = fx + gx * rho + hx * phi + tension * lbmState[IDPHIDX];
        term[IY] = fy + gy * rho + hy * phi + tension * lbmState[IDPHIDY];
        return term;
    }

    // =======================================================
    // correction for the mass conservation
    KOKKOS_INLINE_FUNCTION
    RVect2 force_P(Tag2Quadra tag, LBMState& lbmState) const
    {
        const real_t cs2 = SQR(dx / dt) / e2;
        //~ const real_t rho =lbmState[ID];
        //~ const real_t drhocs2 = cs2*(rho1 - rho0)*SQR(rho)/rho1/rho0;
        const real_t drhocs2 = cs2 * (rhoL - rhoS);
        RVect2 term;
        term[IX] = drhocs2 * lbmState[IDPHIDX];
        term[IY] = drhocs2 * lbmState[IDPHIDY];
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t tau_NS(Tag2Quadra tag, LBMState& lbmState) const
    {
        const real_t phi = lbmState[IPHI];
        const real_t nu = nuL * nuS / (((1.0 - phi) * nuL) + ((phi)*nuS));
        real_t tau = 0.5 + (e2 * nu * dt / SQR(dx));
        return (tau);
    }

}; // end struct model
}; //end namespace
#endif // MODELS_DISSOL_GP_MIXT_H_
