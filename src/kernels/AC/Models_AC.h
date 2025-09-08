#ifndef MODELS_AC_H_
#define MODELS_AC_H_

#include "Index_AC.h"
#include "Kokkos_Macros.hpp"
#include "kernels/Collision_operators.h"
#include "InitConditionsTypes.h"
#include <real_type.h>
namespace PBM_AC {
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
            std::cout << "cs :    " << ((dx / dt) / SQRT(e2)) << std::endl;
            std::cout << "cs2 :    " << SQR(dx / dt) / e2 << std::endl;
            std::cout << "counter_term :    " << counter_term << std::endl;
            std::cout << "MRT_tau0:    " << tauMatrix[0] << std::endl;
            std::cout << "MRT_tau1:    " << tauMatrix[1] << std::endl;
            std::cout << "MRT_tau2:    " << tauMatrix[2] << std::endl;
            std::cout << "MRT_tau3:    " << tauMatrix[3] << std::endl;
            std::cout << "MRT_tau4:    " << tauMatrix[4] << std::endl;
            std::cout << "MRT_tau5:    " << tauMatrix[5] << std::endl;
            std::cout << "MRT_tau6:    " << tauMatrix[6] << std::endl;
            std::cout << "MRT_tau7:    " << tauMatrix[7] << std::endl;
            std::cout << "MRT_tau8:    " << tauMatrix[8] << std::endl;
            std::cout << "init :" << initType << "." << std::endl;
        }
    };

    ModelParams() {};
    ModelParams(const ConfigMap& configMap, LBMParams params)
    {
        time = 0.0;
        dt = params.dt;
        dtprev = dt;
        dx = params.dx;
        e2 = configMap.getFloat("lbm", "e2", 3.0);
        fMach = configMap.getFloat("run", "fMach", 0.04);

        W = configMap.getFloat("params", "W", 0.005);
        Mphi = configMap.getFloat("params", "Mphi", 1.2);
        sigma = configMap.getFloat("params", "sigma", 0.0);

        // Regarder
        PF_advec = configMap.getFloat("params", "PF_advec", 0.0);
        sign = configMap.getFloat("init", "sign", 1);
        counter_term = configMap.getFloat("params", "counter_term", 0.0);

        // Géométrie général
        x0 = configMap.getFloat("init", "x0", 0.0);
        y0 = configMap.getFloat("init", "y0", 0.0);
        z0 = configMap.getFloat("init", "z0", 0.0);
        r0 = configMap.getFloat("init", "r0", 0.2);
        r1 = configMap.getFloat("init", "r1", 0.2);

        // Géométrie Rayleigh-Taylor
        ampl = configMap.getFloat("init", "amplitude", 1.0);
        height = configMap.getFloat("init", "hauteur", 4.0);
        n_wave = configMap.getFloat("init", "nombre_onde", 1.0);
        wave_length = configMap.getFloat("init", "longueur_onde", 1.0);

        // Initialisation Vitesse
        initVX = configMap.getFloat("init", "initVX", 0.0);
        initVY = configMap.getFloat("init", "initVY", 0.0);
        initVZ = configMap.getFloat("init", "initVZ", 0.0);
        U0 = configMap.getFloat("init", "U0", 0.7853975);

        // Initialisation type de la condition initiale
        initType = PHASE_FIELD_INIT_UNDEFINED;
        std::string initTypeStr = std::string(configMap.getString("init", "init_type", "unknown"));

        //! Temps de relaxation MRT
        for (int i = 0; i < NPOP_D3Q27; i++) {
            std::string taui = "tau" + std::to_string(i);
            tauMatrix[i] = configMap.getFloat("MRT", taui, -1.0);
        }

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
        else if (initTypeStr == "shrf_vortex")
            initType = PHASE_FIELD_INIT_SHRF_VORTEX;

        myRank = 0;
#ifdef USE_MPI
        myRank = params.communicator->getRank();
#endif
        showParams();
    }

    //! model params
    real_t W, Mphi, sigma, counter_term;
    real_t dx, dt, dtprev, e2, cs2, fMach;
    real_t PF_advec;
    real_t time;

    //! init params
    real_t x0, y0, z0, r0, r1, sign;
    real_t initVX, initVY, initVZ, U0;
    real_t ampl, height, n_wave, wave_length;

    //! MRT relaxation time
    real_t tauMatrix[27];

    int initType;
    int myRank; // used during mpi simulations to know whether to output params

    KOKKOS_INLINE_FUNCTION real_t phi0(real_t x) const { return 0.5 * (1 + tanh(sign * 2.0 * x / W)); }

    // interpolation functions
    KOKKOS_INLINE_FUNCTION real_t p(real_t phi) const { return SQR(phi) * (3.0 - 2.0 * phi); }
    KOKKOS_INLINE_FUNCTION real_t p_prime(real_t phi) const { return 6.0 * phi * (1.0 - phi); }
    KOKKOS_INLINE_FUNCTION real_t g_prime(real_t phi) const { return 16.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi); }
    KOKKOS_INLINE_FUNCTION real_t h(real_t phi) const { return phi; }
    KOKKOS_INLINE_FUNCTION real_t h_prime(real_t phi) const { return 1.0; }
    KOKKOS_INLINE_FUNCTION real_t q(real_t phi) const { return phi; }
    KOKKOS_INLINE_FUNCTION real_t q_prime(real_t phi) const { return 1.0; }
    KOKKOS_INLINE_FUNCTION real_t aa(real_t phi) const { return 1.0; }

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
    struct Tag2Quadra {
    };

    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(Tag2Quadra type, EquationTag1 tag, const LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }

    // =======================================================
    // first order moment of feq
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        M1(Tag2Quadra type, EquationTag1 tag, const LBMState& lbmState) const
    {
        RVect<dim> term;
        term[IX] = lbmState[IPHI] * lbmState[IU];
        term[IY] = lbmState[IPHI] * lbmState[IV];
        if (dim == 3) {
            term[IZ] = lbmState[IPHI] * lbmState[IW];
        }
        return term;
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
        real_t source_noct = -(1.0 - counter_term) * Mphi / SQR(W) * g_prime(lbmState[IPHI]);
        return (source_noct);
    }

    // =======================================================
    // advection terms for phase field
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        S1(Tag2Quadra type, EquationTag1 tag, const LBMState& lbmState) const
    {
        RVect<dim> term;
        const real_t norm = m_norm(term, lbmState[IDPHIDX], lbmState[IDPHIDY], lbmState[IDPHIDZ]);
        real_t force_ct = Mphi * counter_term * 4.0 * lbmState[IPHI] * (1.0 - lbmState[IPHI]) / W / norm;

        term[IX] = force_ct * lbmState[IDPHIDX]; // + PF_advec*lbmState[IPHI] * lbmState[IU];
        term[IY] = force_ct * lbmState[IDPHIDY]; // + PF_advec*lbmState[IPHI] * lbmState[IV];
        if (dim == 3) {
            term[IZ] = force_ct * lbmState[IDPHIDZ];
        } // + PF_advec*lbmState[IPHI] * lbmState[IW];}
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

}; // end struct model
}
#endif // MODELS_DISSOL_GP_MIXT_H_
