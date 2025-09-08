#ifndef MODELS_AD_H_
#define MODELS_AD_H_
#include "InitConditionsTypes.h"
#include "Index_AD.h"
#include <real_type.h>
namespace PBM_AD {
// ================================================
//
// Models for mixt formulation of grand potential problems
//
// ================================================

struct ModelParams {

    using LBMState = typename Kokkos::Array<real_t, COMPONENT_SIZE>;

    void showParams()
    {
        if (myRank == 0) {
            std::cout << "tau :    " << 0.5 + (e2 * D * dt / SQR(dx)) << std::endl;
            std::cout << "cs :    " << ((dx / dt) / SQRT(e2)) << std::endl;
            std::cout << "cs2 :    " << SQR(dx / dt) / e2 << std::endl;

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
        T = params.tEnd;
        time = 0.0;
        dt = params.dt;
        dtprev = dt;
        dx = params.dx;
        e2 = configMap.getFloat("lbm", "e2", 3.0);
        fMach = configMap.getFloat("run", "fMach", 0.04);

        D = configMap.getFloat("params", "D", 1.2);

        // Géométrie général
        x0 = configMap.getFloat("init", "x0", 0.0);
        y0 = configMap.getFloat("init", "y0", 0.0);
        z0 = configMap.getFloat("init", "z0", 0.0);
        r0 = configMap.getFloat("init", "r0", 0.2);
        r1 = configMap.getFloat("init", "r1", 0.2);
        t1 = configMap.getFloat("init", "t1", 2.0);
        t2 = configMap.getFloat("init", "t2", 4.0);
        period = configMap.getFloat("init", "period", 10.0);
        W = configMap.getFloat("init", "W", 1.0);

        // Initialisation Vitesse
        initVX = configMap.getFloat("init", "initVX", 0.2);
        initVY = configMap.getFloat("init", "initVY", 0.2);
        initVZ = configMap.getFloat("init", "initVZ", 0.2);
        U0 = configMap.getFloat("init", "U0", 0.7853975);

        // Initialisation C
        C0 = configMap.getFloat("init", "C0", 1.0);

        //Initialisation type de la condition initiale
        initType = PHASE_FIELD_INIT_UNDEFINED;
        std::string initTypeStr = std::string(configMap.getString("init", "init_type", "unknown"));

        //! Temps de relaxation MRT
        for (int i = 0; i < NPOP_D3Q27; i++) {
            std::string taui = "tau" + std::to_string(i);
            tauMatrix[i] = configMap.getFloat("MRT", taui, -1.0);
        }

        if (initTypeStr == "AD_rapididité")
            initType = AD_1D_INIT_TUBE;

        myRank = 0;
#ifdef USE_MPI
        myRank = params.communicator->getRank();
#endif
        showParams();
    }

    //! model params
    real_t D;
    real_t dx, dt, dtprev, e2, cs2, fMach;
    real_t time, T, period;

    //! init params
    real_t x0, y0, z0, r0, r1, t1, t2, W;
    real_t initVX, initVY, initVZ, U0;
    real_t C0;

    //!MRT relaxation time
    real_t tauMatrix[27];

    int initType;
    int myRank; //used during mpi simulations to know whether to output params

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
        return lbmState[IC];
    }

    // =======================================================
    // first order moment of feq
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        M1(Tag2Quadra type, EquationTag1 tag, const LBMState& lbmState) const
    {
        RVect<dim> term;
        term[IX] = lbmState[IC] * lbmState[IU];
        term[IY] = lbmState[IC] * lbmState[IV];
        if (dim == 3) {
            term[IZ] = lbmState[IC] * lbmState[IW];
        }
        return term;
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(Tag2Quadra type, EquationTag1 tag, const LBMState& lbmState) const
    {
        return lbmState[IC];
    }

    // =======================================================
    // relaxation coef for LBM scheme of phase field equation
    KOKKOS_INLINE_FUNCTION
    real_t tau(Tag2Quadra type, EquationTag1 tag, const LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 * D * dt / SQR(dx));
        return (tau);
    }

}; // end struct model
}
#endif // MODELS_DISSOL_GP_MIXT_H_
