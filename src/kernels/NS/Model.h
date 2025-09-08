#ifndef MODELS_NS_2AC_C_H_
#define MODELS_NS_2AC_C_H_
#include "InitConditionsTypes.h"
#include "Index.h"
#include <real_type.h>

// ================================================
//
// Models for mixt formulation of grand potential problems
//
// ================================================
namespace PB_NS{
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
            std::cout << "Capillaire = " << SQRT(L * (-gy)) * rho1 * nu1 / Sigma12 << std::endl;
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
        gamma = configMap.getFloat("params", "gamma", 1.0);
        Mphi = configMap.getFloat("params", "Mphi", 1.2);
        // AJOUTS ALAIN
        Sigma12 = configMap.getFloat("params", "Sigma12", 0.0);
        Sigma23 = configMap.getFloat("params", "Sigma23", 0.0);
        Sigma13 = configMap.getFloat("params", "Sigma13", 0.0);
        Sigma12eq = configMap.getFloat("params", "Sigma12eq", 0.0);
        Sigma13eq = configMap.getFloat("params", "Sigma13eq", 0.0);
        Sigma23eq = configMap.getFloat("params", "Sigma23eq", 0.0);
        Gamma1 = Sigma12 + Sigma13 - Sigma23; // Eqs (13) JCP 374 (2018) 668–691
        Gamma2 = Sigma12 + Sigma23 - Sigma13;
        Gamma3 = Sigma13 + Sigma23 - Sigma12;
        GammaT = 3.0 / ((1.0 / Gamma1) + (1.0 / Gamma2) + (1.0 / Gamma3)); // Eq below Eq. (16)
        std::cout << "GammaT   = " << GammaT << std::endl;
        std::cout << "GammaT/2 = " << GammaT / 2.0 << std::endl;
        //std::cout<<"cs2 :    "<<SQR(dx/dt)/e2<<std::endl;
        // FIN AJOUTS ALAIN

        D0 = configMap.getFloat("params", "D0", 1.0);
        D1 = configMap.getFloat("params", "D1", 1.0);
        D2 = configMap.getFloat("params", "D2", 1.0); // AJOUTS ALAIN

        rho0 = configMap.getFloat("params", "rho0", 1.0);
        rho1 = configMap.getFloat("params", "rho1", 1.0);
        rho2 = configMap.getFloat("params", "rho2", 1.0); // AJOUTS ALAIN

        rho0_ini = configMap.getFloat("params", "rho0_ini", 1.0);
        rho0_eq = configMap.getFloat("params", "rho0_eq", 1.0);
        rho1_ini = configMap.getFloat("params", "rho1_ini", 1.0);
        rho1_eq = configMap.getFloat("params", "rho1_eq", 1.0);
        rho2_ini = configMap.getFloat("params", "rho2_ini", 1.0); // AJOUTS ALAIN
        rho2_eq = configMap.getFloat("params", "rho2_eq", 1.0); // AJOUTS ALAIN

        nu0 = configMap.getFloat("params", "nu0", 1.0);
        nu1 = configMap.getFloat("params", "nu1", 1.0);
        nu2 = configMap.getFloat("params", "nu2", 1.0); // AJOUTS ALAIN

        fx = configMap.getFloat("params", "fx", 0.0);
        fy = configMap.getFloat("params", "fy", 0.0);
        fz = configMap.getFloat("params", "fz", 0.0);
        gx = configMap.getFloat("params", "gx", 0.0);
        gy = configMap.getFloat("params", "gy", 0.0);
        gz = configMap.getFloat("params", "gz", 0.0);
        hx = configMap.getFloat("params", "hx", 0.0);
        hy = configMap.getFloat("params", "hy", 0.0);
        hz = configMap.getFloat("params", "hz", 0.0);

        // Composition
        c0_co = configMap.getFloat("params_composition", "c0_co", 0.0);
        c1_co = configMap.getFloat("params_composition", "c1_co", 0.0);
        c2_co = configMap.getFloat("params_composition", "c2_co", 0.0); // AJOUTS ALAIN
        mu_eq = configMap.getFloat("params_composition", "mu_eq", 0.0);
        c0_inf = configMap.getFloat("params_composition", "c0_inf", 0.0);
        c1_inf = configMap.getFloat("params_composition", "c1_inf", 0.0);
        c2_inf = configMap.getFloat("params_composition", "c2_inf", 0.0); // AJOUTS ALAIN

        mu0_inf = c0_inf - c0_co + mu_eq;
        mu1_inf = c1_inf - c1_co + mu_eq;
        mu2_inf = c2_inf - c2_co + mu_eq;

        //!A regarder
        PF_advec = configMap.getFloat("params", "PF_advec", 1.0);
        counter_term = configMap.getFloat("params", "counter_term", 0.0);

        // Géométrie général
        x0 = configMap.getFloat("init", "x0", 0.0);
        y0 = configMap.getFloat("init", "y0", 0.0);
        z0 = configMap.getFloat("init", "z0", 0.0);
        r0 = configMap.getFloat("init", "r0", 0.2);
        r1 = configMap.getFloat("init", "r1", 0.2);

        sign = configMap.getFloat("init", "sign", 1);
        signPhi2 = configMap.getFloat("init", "signPhi2", 1);

        // Géométrie Rayleigh-Taylor
        ampl = configMap.getFloat("init", "amplitude", 1.0);
        height = configMap.getFloat("init", "hauteur", 4.0);
        heightPhi2 = configMap.getFloat("init", "hauteurPhi2", 4.0);
        width = configMap.getFloat("init", "largeur", 4.0);
        n_wave = configMap.getFloat("init", "nombre_onde", 1.0);
        wave_length = configMap.getFloat("init", "longueur_onde", 1.0);

        // Géométrie Two-Plane
        height = configMap.getFloat("init", "hauteur", 4.0);
        initVX_upper = configMap.getFloat("init", "initVX_upper", 4.0);
        initVX_lower = configMap.getFloat("init", "initVX_lower", 4.0);

        // Géométrie Bulles
        U0 = configMap.getFloat("init", "U0", 0.0); // vitesse du vortex
        L0 = configMap.getFloat("init", "L0", 0.0); // Largeur du domaine (info redondante avec xmin et xmax)
        R = configMap.getFloat("init", "rayon", 0.0);
        xC = configMap.getFloat("init", "Xcentre", 0.0);
        yC = configMap.getFloat("init", "Ycentre", 0.0);

        R1 = configMap.getFloat("init", "rayon1", 0.0);
        xC1 = configMap.getFloat("init", "Xcentre1", 0.0);
        yC1 = configMap.getFloat("init", "Ycentre1", 0.0);

        R2 = configMap.getFloat("init", "rayon2", 0.0);
        xC2 = configMap.getFloat("init", "Xcentre2", 0.0);
        yC2 = configMap.getFloat("init", "Ycentre2", 0.0);

        // Valeurs moyennes pour la décomposition spinodale
        phi1moy = configMap.getFloat("init", "phi1moy", 0.0);
        phi2moy = configMap.getFloat("init", "phi2moy", 0.0);
        // Géométrie récipient en U avec épaisseur constante Width_Container
        xmin_Container = configMap.getFloat("init", "xmin_container", 0.0);
        xmax_Container = configMap.getFloat("init", "xmax_container", 0.0);
        ymin_Container = configMap.getFloat("init", "ymin_container", 0.0);
        ymax_Container = configMap.getFloat("init", "ymax_container", 0.0);
        Width_Container = configMap.getFloat("init", "width_container", 0.0);
        //! Temps de relaxation MRT
        for (int i = 0; i < NPOP_D3Q27; i++) {
            std::string taui = "tau" + std::to_string(i);
            tauMatrixAC[i] = configMap.getFloat("MRT_AC", taui, -1.0);
        }

        for (int i = 0; i < NPOP_D3Q27; i++) {
            std::string taui = "tau" + std::to_string(i);
            tauMatrixNS[i] = configMap.getFloat("MRT_NS", taui, -1.0);
        }

        for (int i = 0; i < NPOP_D3Q27; i++) {
            std::string taui = "tau" + std::to_string(i);
            tauMatrixComp[i] = configMap.getFloat("MRT_COMP", taui, -1.0);
        }

        //Initialisation type de la condition initiale
        initType = PHASE_FIELD_INIT_UNDEFINED;
        std::string initTypeStr = std::string(configMap.getString("init", "init_type", "unknown"));

        if (initTypeStr == "vertical") {
            initType = PHASE_FIELD_INIT_VERTICAL;
        } else if (initTypeStr == "sphere") {
            initType = PHASE_FIELD_INIT_SPHERE;
        } else if (initTypeStr == "square") {
            initType = PHASE_FIELD_INIT_SQUARE;
        } else if (initTypeStr == "data") {
            initType = PHASE_FIELD_INIT_DATA;
        } else if (initTypeStr == "2sphere") {
            initType = PHASE_FIELD_INIT_2SPHERE;
        } else if (initTypeStr == "rayleight_taylor") {
            initType = PHASE_FIELD_INIT_TAYLOR;
        } else if (initTypeStr == "two_plane") {
            initType = TWO_PLANE;
        } else if (initTypeStr == "bubble_taylor") {
            initType = BUBBLE_TAYLOR;
        } else if (initTypeStr == "splashing_droplets") {
            initType = THREEPHASES_SPLASHING_DROPLETS;
        } else if (initTypeStr == "rising_droplets") {
            initType = THREEPHASES_RISING_DROPLETS;
        } else if (initTypeStr == "rayleigh_taylor_threephases") {
            initType = THREEPHASES_RAYLEIGH_TAYLOR;
        } else if (initTypeStr == "spinodal_decomposition_threephases") {
            initType = THREEPHASES_SPINODAL_DECOMPOSITION;
        } else if (initTypeStr == "spreading_lens_threephases") {
            initType = THREEPHASES_SPREADING_LENS;
        } else if (initTypeStr == "vortex") {
            initType = PHASE_FIELD_INIT_SHRF_VORTEX;
        } else if (initTypeStr == "threephases_container") {
            initType = THREEPHASES_CONTAINER;
        } else if (initTypeStr == "threephases_capsule") {
            initType = THREEPHASES_CAPSULE;
        }
        myRank = 0;
#ifdef USE_MPI
        myRank = params.communicator->getRank();
#endif
        showParams();
    }

    //! model params
    real_t dx, dt, dtprev, e2, cs2, fMach, L;
    real_t time, T, period;
    real_t W, Mphi, lambda; // Parameters of phase-field equation
    real_t nu1, rho1, nu0, rho0, nu2, rho2; // MODIFS ALAIN
    real_t fx, fy, fz, gx, gy, gz, hx, hy, hz;
    real_t Sigma12, Sigma12eq, Sigma23, Sigma23eq, Sigma13, Sigma13eq;
    real_t Gamma1, Gamma2, Gamma3, GammaT; // MODIFS ALAIN Parameters for surface tensions of 3 phases
    real_t PF_advec;
    real_t gamma, counter_term;
    real_t D0, D1, D2;
    real_t c0_co, c1_co, c2_co, c0_inf, c1_inf, c2_inf, mu_eq, mu0_inf, mu1_inf, mu2_inf;
    real_t rho0_ini, rho0_eq, rho1_ini, rho1_eq, rho2_ini, rho2_eq;
    //! init params
    real_t x0, y0, z0, r0, r1, sign, signPhi2;
    real_t ampl, height, heightPhi2, width, n_wave, wave_length;
    real_t initVX_upper, initVX_lower, U0, L0;
    real_t xC, yC, R, xC1, yC1, R1, xC2, yC2, R2;
    real_t xmin_Container, xmax_Container, ymin_Container, ymax_Container, Width_Container;
    real_t phi1moy, phi2moy;
    //! MRT params
    real_t tauMatrixAC[27], tauMatrixNS[27], tauMatrixComp[27];

    int initType;
    int myRank; //used during mpi simulations to know whether to output params

    // equation of the interface
    KOKKOS_INLINE_FUNCTION real_t phi0(real_t x) const { return 0.5 * (1 + tanh(sign * 2.0 * x / W)); }
    KOKKOS_INLINE_FUNCTION real_t phi2(real_t x) const { return 0.5 * (1 - tanh(sign * 2.0 * x / W)); }
    KOKKOS_INLINE_FUNCTION real_t c0(real_t x) const { return 0.5 * (c1_inf + c0_inf + (c1_inf - c0_inf) * tanh(sign * 2.0 * x / W)); }
    KOKKOS_INLINE_FUNCTION real_t c0_02(real_t x) const { return 0.5 * (c2_inf + c0_inf + (c2_inf - c0_inf) * tanh(sign * 2.0 * x / W)); }
    // interpolation of rho
    KOKKOS_INLINE_FUNCTION real_t hrho(real_t phi) const { return SQR(phi) * (3.0 - 2.0 * phi); }
    KOKKOS_INLINE_FUNCTION real_t hrho_prime(real_t phi) const { return 6.0 * phi * (1.0 - phi); }
    //KOKKOS_INLINE_FUNCTION real_t interp_rho(real_t phi) const {return (1.0-phi) * rho0 + (phi * rho1);}
    KOKKOS_INLINE_FUNCTION real_t rho0c(real_t c) const { return (rho0_eq - rho0_ini) / (c0_co - c0_inf) * (c - c0_inf) + rho0_ini; }
    KOKKOS_INLINE_FUNCTION real_t rho1c(real_t c) const { return (rho1_eq - rho1_ini) / (c1_co - c1_inf) * (c - c1_inf) + rho1_ini; }
    KOKKOS_INLINE_FUNCTION real_t rho2c(real_t c) const { return (rho2_eq - rho2_ini) / (c2_co - c2_inf) * (c - c2_inf) + rho2_ini; }
    KOKKOS_INLINE_FUNCTION real_t rho0c_prime(real_t c) const { return (rho0_eq - rho0_ini) / (c0_co - c0_inf); }
    KOKKOS_INLINE_FUNCTION real_t rho1c_prime(real_t c) const { return (rho1_eq - rho1_ini) / (c1_co - c1_inf); }
    KOKKOS_INLINE_FUNCTION real_t rho2c_prime(real_t c) const { return (rho2_eq - rho2_ini) / (c2_co - c2_inf); }
    KOKKOS_INLINE_FUNCTION real_t interp_rho(real_t phi, real_t c) const { return (1.0 - hrho(phi)) * rho0c(c) + hrho(phi) * rho1c(c); }
    //KOKKOS_INLINE_FUNCTION real_t interp_rho02(real_t phi0,real_t phi2,real_t c) const {return (phi0*rho0c(c) + phi2*rho2c(c));}
    // interpolation functions
    KOKKOS_INLINE_FUNCTION real_t p(real_t phi) const { return SQR(phi) * (3.0 - 2.0 * phi); }
    KOKKOS_INLINE_FUNCTION real_t p_prime(real_t phi) const { return 6.0 * phi * (1.0 - phi); }
    KOKKOS_INLINE_FUNCTION real_t g_prime(real_t phi) const { return 16.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi); }
    KOKKOS_INLINE_FUNCTION real_t g2_prime(real_t phi) const { return phi * (1.0 - phi) * (1.0 - 2.0 * phi); } // Ajout Alain
    // interpolation function for c_co
    KOKKOS_INLINE_FUNCTION real_t h(real_t phi) const { return phi; }
    KOKKOS_INLINE_FUNCTION real_t h_prime(real_t phi) const { return 1.0; }
    KOKKOS_INLINE_FUNCTION real_t c_co(real_t phi) const { return c1_co * h(phi) + c0_co * (1 - h(phi)); }
    // Echange entre phase 0 et phase 2               AJOUT ALAIN
    KOKKOS_INLINE_FUNCTION real_t c_co_Exchange02(real_t phi0, real_t phi2) const { return c0_co * phi0 + c2_co * phi2; }
    KOKKOS_INLINE_FUNCTION real_t interpol_D02(real_t phi0, real_t phi2) const { return D0 * phi0 + D2 * phi2; }
    // ingerpolation function for D0 and D1
    KOKKOS_INLINE_FUNCTION real_t q(real_t phi) const { return phi; }
    KOKKOS_INLINE_FUNCTION real_t q_prime(real_t phi) const { return 1.0; }
    KOKKOS_INLINE_FUNCTION real_t interpol_D(real_t phi) const { return D1 * q(phi) + D0 * (1 - q(phi)); }
    // AJOUT ALAIN POUR MODIFIER LA TENSION DE SURFACE Sigma12 en fct de c
    KOKKOS_INLINE_FUNCTION real_t Sigma12c(real_t c) const { return (Sigma12eq - Sigma12) / (c2_co - c2_inf) * (c - c2_inf) + Sigma12; }
    KOKKOS_INLINE_FUNCTION real_t Sigma13c(real_t c) const { return (Sigma13eq - Sigma13) / (c1_co - c1_inf) * (c - c1_inf) + Sigma13; }
    KOKKOS_INLINE_FUNCTION real_t Sigma23c(real_t c) const { return (Sigma23eq - Sigma23) / (c1_co - c1_inf) * (c - c1_inf) + Sigma23; }

    // ================================================
    //
    // functions for Fakhari et. al.'s model
    //
    // ================================================

    struct TagBase {
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
    RVect2 S_ct(const LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY])) + NORMAL_EPSILON;
        real_t force_ct = Mphi * counter_term * 4.0 * lbmState[IPHI] * (1.0 - lbmState[IPHI]) / W / norm;
        RVect2 term;
        term[IX] = force_ct * lbmState[IDPHIDX];
        term[IY] = force_ct * lbmState[IDPHIDY];
        return term;
    }

    // Coupling term with lambda
    KOKKOS_INLINE_FUNCTION
    real_t S_st(EquationTag1 tag, const LBMState& lbmState) const
    {
        real_t S = p_prime(lbmState[IPHI]) * (c0_co - c1_co) * (mu(lbmState[IC], lbmState[IPHI]) - mu_eq);
        return -lambda * Mphi / SQR(W) * S;
    }
    // AJOUT ALAIN POUR EFFET DU TRANSFERT COMPOSITION ENTRE 1 ET 2 sur
    KOKKOS_INLINE_FUNCTION
    real_t S_st012(EquationTag1 tag, const LBMState& lbmState) const
    {
        real_t valmu = mu012(lbmState[IC], lbmState[IPHI3], lbmState[IPHI], lbmState[IPHI2]);
        real_t S = p_prime(lbmState[IPHI]) * (c1_co - c2_co) * (valmu - mu_eq);
        return -lambda * (Mphi / SQR(W)) * S;
    }

    // =======================================================
    // 					Model for phase field 2 AJOUTS ET MODIFS ALAIN
    // =======================================================

    // =======================================================
    // relaxation coef for LBM scheme of phase field equation
    KOKKOS_INLINE_FUNCTION
    real_t tau_PHI2(EquationTag4 tag, const LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 * Mphi * dt / SQR(dx));
        return (tau);
    }

    // =======================================================
    // zero order moment for phase field 2
    KOKKOS_INLINE_FUNCTION
    real_t M0_PHI2(EquationTag4 tag, const LBMState& lbmState) const
    {
        return lbmState[IPHI2];
    }

    // =======================================================
    // first order moment for phase field 2
    KOKKOS_INLINE_FUNCTION
    RVect2 M1_PHI2(EquationTag4 tag, const LBMState& lbmState) const
    {
        RVect2 term;
        term[IX] = lbmState[IPHI2] * lbmState[IU];
        term[IY] = lbmState[IPHI2] * lbmState[IV];
        return term;
    }

    // =======================================================
    // second order moment for phase field 2
    KOKKOS_INLINE_FUNCTION
    real_t M2_PHI2(EquationTag4 tag, const LBMState& lbmState) const
    {
        return lbmState[IPHI2];
    }

    // =======================================================
    // Sources term for phase field 2 and counter term for phi3
    // Derivative of double-well
    KOKKOS_INLINE_FUNCTION
    real_t S2_dw(const LBMState& lbmState) const
    {
        real_t source_noct = -(1.0 - counter_term) * Mphi / SQR(W) * 16.0 * lbmState[IPHI2] * (1.0 - lbmState[IPHI2]) * (1.0 - 2.0 * lbmState[IPHI2]);
        return source_noct;
    }
    // Computation of macroscopic counter term with Phi2 (vector)
    KOKKOS_INLINE_FUNCTION
    RVect2 S2_ct(const LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHI2DX]) + SQR(lbmState[IDPHI2DY])) + NORMAL_EPSILON;
        real_t force_ct = Mphi * counter_term * 4.0 * lbmState[IPHI2] * (1.0 - lbmState[IPHI2]) / W / norm;
        RVect2 term;
        term[IX] = force_ct * lbmState[IDPHI2DX];
        term[IY] = force_ct * lbmState[IDPHI2DY];
        return term;
    }
    // Coupling with thermodynamic (equilibirum compositions)
    KOKKOS_INLINE_FUNCTION
    real_t S2_st(const LBMState& lbmState) const
    {
        real_t S = p_prime(lbmState[IPHI]) * (c0_co - c1_co) * (mu(lbmState[IC], lbmState[IPHI]) - mu_eq);
        return -lambda * Mphi / SQR(W) * S;
    }

    // Computation of macroscopic counter term with Phi3 (vector) for Eq phi and Eq phi2
    KOKKOS_INLINE_FUNCTION
    RVect2 S3_ct(const LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHI3DX]) + SQR(lbmState[IDPHI3DY])) + NORMAL_EPSILON;
        real_t force_ct = Mphi * counter_term * 4.0 * lbmState[IPHI3] * (1.0 - lbmState[IPHI3]) / W / norm;
        RVect2 term;
        term[IX] = force_ct * lbmState[IDPHI3DX];
        term[IY] = force_ct * lbmState[IDPHI3DY];
        return term;
    }

    // =======================================================
    // 					Model for Navier-Stokes
    // =======================================================

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t tau_NS(const LBMState& lbmState) const
    {
        const real_t phi = lbmState[IPHI];
        const real_t nu = nu0 * nu1 / (((1.0 - phi) * nu1) + ((phi)*nu0));
        real_t tau = 0.5 + (e2 * nu * dt / SQR(dx));
        return (tau);
    }
    // relaxation coef for LBM scheme of Navier-Stokes equation for three phases
    KOKKOS_INLINE_FUNCTION
    real_t tau_3phases_NS(const LBMState& lbmState) const
    {
        const real_t tau1 = nu1 * e2 * dt / SQR(dx);
        const real_t tau2 = nu2 * e2 * dt / SQR(dx);
        const real_t tau3 = nu0 * e2 * dt / SQR(dx);
        const real_t phi1 = lbmState[IPHI];
        const real_t phi2 = lbmState[IPHI2];
        const real_t phi3 = lbmState[IPHI3];
        //real_t tau = tau1*phi1 + tau2*phi2 + tau3*phi3;
        real_t tau = 0.5 + (tau1 * tau2 * tau3) / (tau2 * tau3 * phi1 + tau1 * tau3 * phi2 + tau1 * tau2 * phi3);
        //real_t Heaviside1, Heaviside2 ;
        //Heaviside1 = 0.5*(1.0+tanh((phi1-0.5)/0.12)) ;
        //Heaviside2 = 0.5*(1.0+tanh((phi2-0.5)/0.12)) ;
        //real_t tau = 0.5 + (tau1-tau3)*Heaviside1 + (tau2-tau3)*Heaviside2 + tau3 ;

        return (tau);
    }
    // =======================================================
    // Pressure term for NS
    KOKKOS_INLINE_FUNCTION
    RVect2 force_P2(const LBMState& lbmState) const
    {
        const RVect2 gradrho = grad_rho(lbmState);
        const real_t scal = lbmState[IP] / lbmState[ID];
        RVect2 term;
        term[IX] = -scal * gradrho[IX];
        term[IY] = -scal * gradrho[IY];
        return term;
    }
    // ===================== ALAIN ===========================
    // Pressure term for NS (version with gradrho calculated numerically)
    KOKKOS_INLINE_FUNCTION
    RVect2 force_P_rhonum(const LBMState& lbmState) const
    {
        //const RVect2 gradrho = grad_rho(lbmState);
        RVect2 gradrho;
        gradrho[IX] = lbmState[IDRHODX];
        gradrho[IY] = lbmState[IDRHODY];
        const real_t scal = lbmState[IP] / lbmState[ID];
        RVect2 term;
        term[IX] = -scal * gradrho[IX];
        term[IY] = -scal * gradrho[IY];
        return term;
    }
    // =======================================================
    // Body term for NS
    KOKKOS_INLINE_FUNCTION
    RVect2 force_G(const LBMState& lbmState) const
    {
        RVect2 term;
        term[IX] = gx * lbmState[ID];
        term[IY] = gy * lbmState[ID];
        return term;
    }

    // =======================================================
    // Tension superficiel term for NS (two-phase case)
    KOKKOS_INLINE_FUNCTION
    RVect2 force_TS(const LBMState& lbmState) const
    {
        const real_t tension = Sigma12 * 1.5 / W * (g_prime(lbmState[IPHI]) - SQR(W) * lbmState[ILAPLAPHI]);
        RVect2 term;
        term[IX] = tension * lbmState[IDPHIDX];
        term[IY] = tension * lbmState[IDPHIDY];
        return term;
    }

    // =================== AJOUT ALAIN ====================================
    // Surface tension term number 1 for NS: three-phase case (implementation of Eq. (16) in Ref)
    KOKKOS_INLINE_FUNCTION
    RVect2 force_TS1(const LBMState& lbmState) const
    {
        real_t c = lbmState[IC];
        real_t Gamma1 = Sigma12c(c) + Sigma13 - Sigma23;
        real_t Gamma2 = Sigma12c(c) + Sigma23 - Sigma13;
        real_t Gamma3 = Sigma13 + Sigma23 - Sigma12c(c);
        real_t GammaT = 3.0 / ((1.0 / Gamma1) + (1.0 / Gamma2) + (1.0 / Gamma3));
        const real_t factor = 4.0 * GammaT / W;
        const real_t term1 = (1.0 / Gamma2) * ((12.0 * Gamma1 / W) * g2_prime(lbmState[IPHI]) - (12.0 * Gamma2 / W) * g2_prime(lbmState[IPHI2]));
        const real_t term2 = (1.0 / Gamma3) * ((12.0 * Gamma1 / W) * g2_prime(lbmState[IPHI]) - (12.0 * Gamma3 / W) * g2_prime(lbmState[IPHI3]));
        const real_t mu1_pot = factor * (term1 + term2) - (3.0 / 4.0) * W * Gamma1 * lbmState[ILAPLAPHI];
        RVect2 term;
        term[IX] = mu1_pot * lbmState[IDPHIDX];
        term[IY] = mu1_pot * lbmState[IDPHIDY];
        return term;
    }
    // =================== AJOUT ALAIN ====================================
    // Surface tension term number 2 for NS for phi2 (AJOUT ALAIN)
    KOKKOS_INLINE_FUNCTION
    RVect2 force_TS2(const LBMState& lbmState) const
    {
        real_t c = lbmState[IC];
        real_t Gamma1 = Sigma12c(c) + Sigma13 - Sigma23;
        real_t Gamma2 = Sigma12c(c) + Sigma23 - Sigma13;
        real_t Gamma3 = Sigma13 + Sigma23 - Sigma12c(c);
        real_t GammaT = 3.0 / ((1.0 / Gamma1) + (1.0 / Gamma2) + (1.0 / Gamma3));
        const real_t factor = 4.0 * GammaT / W;
        const real_t term1 = (1.0 / Gamma1) * ((12.0 * Gamma2 / W) * g2_prime(lbmState[IPHI2]) - (12.0 * Gamma1 / W) * g2_prime(lbmState[IPHI]));
        const real_t term2 = (1.0 / Gamma3) * ((12.0 * Gamma2 / W) * g2_prime(lbmState[IPHI2]) - (12.0 * Gamma3 / W) * g2_prime(lbmState[IPHI3]));
        const real_t mu2_pot = factor * (term1 + term2) - (3.0 / 4.0) * W * Gamma2 * lbmState[ILAPLAPHI2];
        RVect2 term;
        term[IX] = mu2_pot * lbmState[IDPHI2DX];
        term[IY] = mu2_pot * lbmState[IDPHI2DY];
        return term;
    }
    // =================== AJOUT ALAIN ====================================
    // Surface tension term number 3 for NS for phi3 (AJOUT ALAIN)
    KOKKOS_INLINE_FUNCTION
    RVect2 force_TS3(const LBMState& lbmState) const
    {
        real_t c = lbmState[IC];
        real_t Gamma1 = Sigma12c(c) + Sigma13 - Sigma23;
        real_t Gamma2 = Sigma12c(c) + Sigma23 - Sigma13;
        real_t Gamma3 = Sigma13 + Sigma23 - Sigma12c(c);
        real_t GammaT = 3.0 / ((1.0 / Gamma1) + (1.0 / Gamma2) + (1.0 / Gamma3));
        const real_t factor = 4.0 * GammaT / W;
        const real_t term1 = (1.0 / Gamma1) * ((12.0 * Gamma3 / W) * g2_prime(lbmState[IPHI3]) - (12.0 * Gamma1 / W) * g2_prime(lbmState[IPHI]));
        const real_t term2 = (1.0 / Gamma2) * ((12.0 * Gamma3 / W) * g2_prime(lbmState[IPHI3]) - (12.0 * Gamma2 / W) * g2_prime(lbmState[IPHI2]));
        const real_t mu3_pot = factor * (term1 + term2) - (3.0 / 4.0) * W * Gamma3 * lbmState[ILAPLAPHI3];
        RVect2 term;
        term[IX] = mu3_pot * lbmState[IDPHI3DX];
        term[IY] = mu3_pot * lbmState[IDPHI3DY];
        return term;
    }
    // =================== AJOUT ALAIN ====================================
    // Force Flux nul pour une surface solide (AJOUT ALAIN)
    KOKKOS_INLINE_FUNCTION
    RVect2 force_Beckermann(const LBMState& lbmState) const
    {
        real_t coeff_h, phi1, factor, factor0, factor2;
        coeff_h = 2.757;
        phi1 = lbmState[IPHI];
        factor0 = (nu0 * rho0) * coeff_h * phi1 * phi1 * (1.0 - phi1) / (W * W);
        factor2 = (nu2 * rho2) * coeff_h * phi1 * phi1 * (1.0 - phi1) / (W * W);
        factor = factor0 + factor2;
        RVect2 term;
        term[IX] = factor * lbmState[IU];
        term[IY] = factor * lbmState[IV];
        return term;
    }
    // =======================================================
    // Grad rho computed by the chain rule --> (grad phi) and (grad c) // Comments ALAIN
    KOKKOS_INLINE_FUNCTION
    RVect2 grad_rho(const LBMState& lbmState) const
    {
        RVect2 term;
        real_t phi = lbmState[IPHI];
        real_t c = lbmState[IC];
        real_t drhodphi = (rho1c(lbmState[IC]) - rho0c(lbmState[IC])) * hrho_prime(lbmState[IPHI]);
        real_t drhodc = (1 - hrho(phi)) * rho0c_prime(c) + hrho(phi) * rho1c_prime(c);
        term[IX] = drhodphi * lbmState[IDPHIDX] + drhodc * lbmState[IDCDX];
        term[IY] = drhodphi * lbmState[IDPHIDY] + drhodc * lbmState[IDCDY];
        return term;
    }

    // =======================================================
    // 					Model for composition
    // =======================================================

    KOKKOS_INLINE_FUNCTION
    real_t mu(real_t c, real_t phi) const
    {
        return mu_eq + c - c_co(phi);
    }
    //	AJOUT ALAIN POUR ECHANGE ENTRE PHASE 0 ET PHASE 2
    KOKKOS_INLINE_FUNCTION
    real_t mu012(real_t c, real_t phi0, real_t phi1, real_t phi2) const
    {
        return mu_eq + c - (c0_co * phi0 + c1_co * phi1 + c2_co * phi2);
    }
    //	FIN AJOUT
    KOKKOS_INLINE_FUNCTION
    real_t tau_C(TagBase tag, const LBMState& lbmState) const
    {
        return 0.5 + (e2 * dt / (SQR(dx) * gamma));
    }
    //	AJOUT ALAIN
    KOKKOS_INLINE_FUNCTION
    real_t tau_C02(TagBase tag, const LBMState& lbmState) const
    {
        return 0.5 + (e2 * dt * interpol_D02(lbmState[IPHI3], lbmState[IPHI2]) / (SQR(dx)));
    }
    // relaxation coef for LBM scheme of Navier-Stokes equation for three phases
    KOKKOS_INLINE_FUNCTION
    real_t tau_3phases_Compos(const LBMState& lbmState) const
    {
        const real_t tau1 = D1 * e2 * dt / SQR(dx);
        const real_t tau2 = D2 * e2 * dt / SQR(dx);
        const real_t tau3 = D0 * e2 * dt / SQR(dx);
        const real_t phi1 = lbmState[IPHI];
        const real_t phi2 = lbmState[IPHI2];
        const real_t phi3 = lbmState[IPHI3];
        real_t tau = 0.5 + (tau1 * tau2 * tau3) / (tau2 * tau3 * phi1 + tau1 * tau3 * phi2 + tau1 * tau2 * phi3);
        //real_t Heaviside1, Heaviside2 ;
        //Heaviside1 = 0.5*(1.0+tanh((phi1-0.5)/0.12)) ;
        //Heaviside2 = 0.5*(1.0+tanh((phi2-0.5)/0.12)) ;
        //real_t tau = 0.5 + (tau1-tau3)*Heaviside1 + (tau2-tau3)*Heaviside2 + tau3 ;

        return (tau);
    }
    //	FIN AJOUT ALAIN
    KOKKOS_INLINE_FUNCTION
    real_t M0_C(const LBMState& lbmState) const
    {
        return lbmState[IC];
    }

    KOKKOS_INLINE_FUNCTION
    RVect2 M1_C(const LBMState& lbmState) const
    {
        RVect2 term;
        term[IX] = lbmState[IC] * lbmState[IU];
        term[IY] = lbmState[IC] * lbmState[IV];
        return term;
    }

    KOKKOS_INLINE_FUNCTION
    real_t M2_C(TagBase tag, const LBMState& lbmState) const
    {
        return gamma * interpol_D(lbmState[IPHI]) * lbmState[IMU];
    }

    KOKKOS_INLINE_FUNCTION
    RVect2 S0_C(TagBase tag, const LBMState& lbmState) const
    {
        RVect2 term;
        real_t coeff = (D1 - D0) * q_prime(lbmState[IPHI]) * lbmState[IMU];
        term[IX] = coeff * lbmState[IDPHIDX];
        term[IY] = coeff * lbmState[IDPHIDY];
        return term;
    }

}; // end struct model
}
#endif
