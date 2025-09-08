#ifndef MODELS_NS_PF_F_H_
#define MODELS_NS_PF_F_H_
#include "InitConditionsTypes.h"
#include "Index_NS_AC_Fakhari.h"
#include <real_type.h>

namespace PBM_NS_AC_Fakhari {

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
        if (myRank == 0) {
            std::cout << "W :    " << W << std::endl;
            std::cout << "tauphi :    " << 0.5 + (e2 * Mphi * dt / SQR(dx)) << std::endl;
            std::cout << "tauNS :    " << (0.5 + (e2 * nu0 * dt / SQR(dx))) << std::endl;
            std::cout << "cs :    " << ((dx / dt) / SQRT(e2)) << std::endl;
            std::cout << "cs2 :    " << SQR(dx / dt) / e2 << std::endl;
            std::cout << "counter_term :    " << counter_term << std::endl;
            std::cout << "init :" << initType << "." << std::endl;
            std::cout << "Reynolds = " << SQRT(L * (-gy)) * L / nu0 << std::endl;
            std::cout << "Peclet = " << SQRT(L * (-gy)) * L / Mphi << std::endl;
            std::cout << "Capillaire = " << SQRT(L * (-gy)) * rho1 * nu1 / sigma << std::endl;
        }
    };

    ModelParams() {};
    ModelParams(const ConfigMap& configMap, LBMParams params)
    {
        T = params.tEnd;
        time = 0.0;
        dt = params.dt;
        dtprev = dt;
        dx = params.dx;
        e2 = configMap.getFloat("lbm", "e2", 3.0);
        fMach = configMap.getFloat("run", "fMach", 0.04);

        L = configMap.getFloat("mesh", "xmax", 1.0) - configMap.getFloat("mesh", "xmin", 0.0);

        W = configMap.getFloat("params", "W", 0.005);
        Mphi = configMap.getFloat("params", "Mphi", 1.2);
        sigma = configMap.getFloat("params", "sigma", 0.0);

        rho0 = configMap.getFloat("params", "rho0", 1.0);
        rho1 = configMap.getFloat("params", "rho1", 1.0);

        nu0 = configMap.getFloat("params", "nu0", 1.0);
        nu1 = configMap.getFloat("params", "nu1", 1.0);

        fx = configMap.getFloat("params", "fx", 0.0);
        fy = configMap.getFloat("params", "fy", 0.0);
        fz = configMap.getFloat("params", "fz", 0.0);
        gx = configMap.getFloat("params", "gx", 0.0);
        gy = configMap.getFloat("params", "gy", 0.0);
        gz = configMap.getFloat("params", "gz", 0.0);
        hx = configMap.getFloat("params", "hx", 0.0);
        hy = configMap.getFloat("params", "hy", 0.0);
        hz = configMap.getFloat("params", "hz", 0.0);

        //!A regarder
        PF_advec = configMap.getFloat("params", "PF_advec", 0.0);
        counter_term = configMap.getFloat("params", "counter_term", 0.0);

        // Géométrie général
        x0 = configMap.getFloat("init", "x0", 0.0);
        y0 = configMap.getFloat("init", "y0", 0.0);
        z0 = configMap.getFloat("init", "z0", 0.0);
        r0 = configMap.getFloat("init", "r0", 0.2);
        r1 = configMap.getFloat("init", "r1", 0.2);

        sign = configMap.getFloat("init", "sign", 1);

        // Initialisation Vitesse
        initVX = configMap.getFloat("init", "initVX", 0.2);
        initVY = configMap.getFloat("init", "initVY", 0.2);
        initVZ = configMap.getFloat("init", "initVZ", 0.2);

        // Géométrie Rayleigh-Taylor
        ampl = configMap.getFloat("init", "amplitude", 1.0);
        height = configMap.getFloat("init", "hauteur", 4.0);
        n_wave = configMap.getFloat("init", "nombre_onde", 1.0);
        wave_length = configMap.getFloat("init", "longueur_onde", 1.0);

        //! Temps de relaxation MRT
        for (int i = 0; i < NPOP_D3Q27; i++) {
            std::string taui = "tau" + std::to_string(i);
            tauMatrixAC[i] = configMap.getFloat("MRT_AC", taui, -1.0);
        }

        for (int i = 0; i < NPOP_D3Q27; i++) {
            std::string taui = "tau" + std::to_string(i);
            tauMatrixNS[i] = configMap.getFloat("MRT_NS", taui, -1.0);
        }

        //Initialisation type de la condition initiale
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
        else if (initTypeStr == "2sphere")
            initType = PHASE_FIELD_INIT_2SPHERE;
        else if (initTypeStr == "rayleight_taylor")
            initType = PHASE_FIELD_INIT_TAYLOR;

        myRank = 0;
#ifdef USE_MPI
        myRank = params.communicator->getRank();
#endif
        showParams();
    }

    //! model params
    real_t W, Mphi, counter_term;
    real_t dx, dt, dtprev, e2, cs2, fMach, L;
    real_t time, T;
    real_t nu1, rho1, nu0, rho0;
    real_t fx, fy, fz, gx, gy, gz, hx, hy, hz, sigma;
    real_t PF_advec;
    //! init params
    real_t x0, y0, z0, r0, r1, sign;
    real_t initVX, initVY, initVZ;
    real_t ampl, height, n_wave, wave_length;
    //! MRT params
    real_t tauMatrixAC[27], tauMatrixNS[27];

    int initType;
    int myRank; //used during mpi simulations to know whether to output params

    // equation of the interface
    KOKKOS_INLINE_FUNCTION real_t phi0(real_t x) const { return 0.5 * (1 + tanh(sign * 2.0 * x / W)); }
    // interpolation for rho
    KOKKOS_INLINE_FUNCTION real_t interp_rho(real_t phi) const { return (1.0 - phi) * rho0 + (phi * rho1); }
    // interpolation for phi
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
    // functions for Fakhari et. al.'s model
    //
    // ================================================

    struct TagFakhari {
    };

    // =======================================================
    // 					Model for phase field
    // =======================================================

    // =======================================================
    // relaxation coef for LBM scheme of phase field equation
    KOKKOS_INLINE_FUNCTION
    real_t tau_PHI(EquationTag1 tag, const LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 * Mphi * dt / SQR(dx));
        return (tau);
    }

    // =======================================================
    // zero order moment for phase field
    KOKKOS_INLINE_FUNCTION
    real_t M0_PHI(EquationTag1 tag, const LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }

    // =======================================================
    // first order moment for phase field
    KOKKOS_INLINE_FUNCTION
    RVect2 M1_PHI(EquationTag1 tag, const LBMState& lbmState) const
    {
        RVect2 term;
        term[IX] = lbmState[IPHI] * lbmState[IU];
        term[IY] = lbmState[IPHI] * lbmState[IV];
        return term;
    }

    // =======================================================
    // second order moment for phase field
    KOKKOS_INLINE_FUNCTION
    real_t M2_PHI(EquationTag1 tag, const LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }

    // =======================================================
    // Sources term for phase field
    KOKKOS_INLINE_FUNCTION
    real_t S_dw(EquationTag1 tag, const LBMState& lbmState) const
    {
        real_t source_noct = -(1.0 - counter_term) * Mphi / SQR(W) * 16.0 * lbmState[IPHI] * (1.0 - lbmState[IPHI]) * (1.0 - 2.0 * lbmState[IPHI]);
        return source_noct;
    }

    KOKKOS_INLINE_FUNCTION
    RVect2 S_ct(EquationTag1 tag, const LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY])) + NORMAL_EPSILON;
        real_t force_ct = Mphi * counter_term * 4.0 * lbmState[IPHI] * (1.0 - lbmState[IPHI]) / W / norm;
        RVect2 term;
        term[IX] = force_ct * lbmState[IDPHIDX];
        term[IY] = force_ct * lbmState[IDPHIDY];
        return term;
    }

    // =======================================================
    // 					Model for Navier-Stokes
    // =======================================================

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t tau_NS(TagFakhari tag, const LBMState& lbmState) const
    {
        const real_t phi = lbmState[IPHI];
        const real_t mu = nu0 * rho0 + phi * (nu1 * rho1 - nu0 * rho0);
        real_t tau = 0.5 + (e2 * mu * dt / (SQR(dx) * lbmState[ID]));
        return (tau);
    }

    // =======================================================
    // Pressure term for NS
    KOKKOS_INLINE_FUNCTION
    RVect2 force_P(TagFakhari tag, const LBMState& lbmState) const
    {
        const real_t scal = (rho1 - rho0) * lbmState[IP] / lbmState[ID];
        RVect2 term;
        term[IX] = -scal * lbmState[IDPHIDX];
        term[IY] = -scal * lbmState[IDPHIDY];
        return term;
    }

    // =======================================================
    // Body term for NS
    KOKKOS_INLINE_FUNCTION
    RVect2 force_G(TagFakhari tag, const LBMState& lbmState) const
    {
        RVect2 term;
        term[IX] = gx * lbmState[ID];
        term[IY] = gy * lbmState[ID];
        return term;
    }

    // =======================================================
    // Tension superficiel term for NS
    KOKKOS_INLINE_FUNCTION
    RVect2 force_TS(TagFakhari tag, const LBMState& lbmState) const
    {
        const real_t tension = sigma * 1.5 / W * (g_prime(lbmState[IPHI]) - SQR(W) * lbmState[ILAPLAPHI]);
        RVect2 term;
        term[IX] = tension * lbmState[IDPHIDX];
        term[IY] = tension * lbmState[IDPHIDY];
        return term;
    }

}; // end struct model
}
#endif
