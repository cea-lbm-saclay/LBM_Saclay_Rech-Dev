#ifndef LBMSCHEME_NS_2AC_C_3D_H_
#define LBMSCHEME_NS_2AC_C_3D_H_

#include <limits> // for std::numeric_limits

#ifdef __CUDA_ARCH__
#include <math_constants.h>
#endif // __CUDA_ARCH__

#include "FieldManager.h"
#include "LBMParams.h"
#include "LBM_Base_Functor.h"

#include "Models_NS_2AC_Comp3D.h"

namespace PBM_NS_2AC_Compo3D {
template <int dim, int npop, typename modelType>
struct LBMScheme : public LBMSchemeBase<dim, npop> {
    using Base = LBMSchemeBase<dim, npop>;
    using LBMState = typename Kokkos::Array<real_t, COMPONENT_SIZE>;
    using id2index_t = Kokkos::Array<int, COMPONENT_SIZE>;
    using LBMArray = typename LBMBaseFunctor<dim, npop>::LBMArray;
    using FArray = typename LBMBaseFunctor<dim, npop>::FArray;
    using LBM_Weights = typename LBMBaseFunctor<dim, npop>::LBM_Weights;
    using LBM_speeds = typename LBMBaseFunctor<dim, npop>::LBM_speeds;
    using LBM_speeds_opposite = typename LBMBaseFunctor<dim, npop>::LBM_speeds_opposite;
    using FState = typename Kokkos::Array<real_t, npop>;

    static constexpr real_t NORMAL_EPSILON = 1.0e-16;

    using BGK_Collider = BGKCollider<dim, npop>;
    using MRT_Collider = MRTCollider<dim, npop>;

    ModelParams Model;
    modelType type;
    LBM_speeds E;
    LBM_speeds_opposite Ebar;
    LBM_Weights w;
    EquationTag1 tagPHI;
    EquationTag2 tagNS;
    EquationTag3 tagCOMP;
    EquationTag4 tagPHI2; // AJOUT ALAIN

    LBMScheme()
        : LBMSchemeBase<dim, npop>()
        , E(LBMBaseFunctor<dim, npop>::E)
        , Ebar(LBMBaseFunctor<dim, npop>::Ebar)
        , w(LBMBaseFunctor<dim, npop>::w) {};

    LBMScheme(ConfigMap configMap, LBMParams params, LBMArray& lbm_data)
        : LBMSchemeBase<dim, npop>(params, lbm_data)
        , Model(configMap, params)
        , E(LBMBaseFunctor<dim, npop>::E)
        , Ebar(LBMBaseFunctor<dim, npop>::Ebar)
        , w(LBMBaseFunctor<dim, npop>::w) {};

    /*! ===================================================================
	Organisation of the scheme
	- setup_collider :
		* one for each equation
		* one for each Collider (BGK or MRT)
	- make_boundary
	- init_macro
	- update_macro
	- update_macro_grad
    ==================================================================== */

    // =====================================================================
    //						SETUP COLLIDER
    // =====================================================================

    // ==================== PHASE FIELD EQUATION 1 ===========================

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag1 tag, const IVect<dim>& IJK, BGK_Collider& collider) const
    {
        //std::cout<<"Je suis dans setup_collider Eq 1"<< std::endl;
        const real_t dt = Model.dt;
        const real_t dx = Model.dx;
        const real_t e2 = Model.e2;
        const real_t cs2 = 1.0 / e2 * SQR(dx / dt);

        LBMState lbmState;
        Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmState);

        const real_t M0 = Model.M0_PHI(tag, lbmState);
        const RVect<dim> M1 = Model.M1_PHI<dim>(tag, lbmState);
        const real_t M2 = Model.M2_PHI(tag, lbmState);
        //const real_t G_st = Model.S_st(tag, lbmState);
        const real_t G_st = Model.S_phase_change_phi1(lbmState);

        // AJOUTS ALAIN pour counter terms phi2 et phi (ATTENTION pas de tag en argument)
        const real_t G_dw = Model.S_dw(tag, lbmState);
        const RVect<dim> G1_ct = Model.S_ct<dim>(lbmState);
        const RVect<dim> G2_ct = Model.S2_ct<dim>(lbmState);
        const RVect<dim> G3_ct = Model.S3_ct<dim>(lbmState);
        RVect<dim> G123_ct;
        G123_ct[IX] = (2.0 / 3.0) * G1_ct[IX] - (1.0 / 3.0) * G2_ct[IX] - (1.0 / 3.0) * G3_ct[IX];
        G123_ct[IY] = (2.0 / 3.0) * G1_ct[IY] - (1.0 / 3.0) * G2_ct[IY] - (1.0 / 3.0) * G3_ct[IY];
        G123_ct[IZ] = (2.0 / 3.0) * G1_ct[IZ] - (1.0 / 3.0) * G2_ct[IZ] - (1.0 / 3.0) * G3_ct[IZ];
        // FIN AJOUTS ALAIN

        // compute collision rate
        collider.tau = Model.tau_PHI(tag, lbmState);
        real_t staudx = 1 / ((collider.tau - 0.5) * dx / e2);
        // ipop = 0
        collider.f[0] = Base::get_f_val(tag, IJK, 0);
        collider.S0[0] = w[0] * dt * (G_st + G_dw + staudx * Base::compute_scal(0, G123_ct)); // MODIF ALAIN
        collider.feq[0] = M0 - (1 - w[0]) * M2 - 0.5 * collider.S0[0];

        // ipop > 0
        for (int ipop = 1; ipop < npop; ++ipop) {
            collider.f[ipop] = this->get_f_val(tag, IJK, ipop);
            collider.S0[ipop] = w[ipop] * dt * (G_st + G_dw + staudx * Base::compute_scal(ipop, G123_ct)); // MODIF ALAIN
            collider.feq[ipop] = w[ipop] * (M2 + Model.PF_advec * dx / dt * Base::compute_scal(ipop, M1) / cs2) - 0.5 * (collider.S0[ipop]);
        }
    } // end of setup_collider for phase field equation

    // ==================== PHASE FIELD EQUATION 2 (AJOUTS ALAIN) ===========================

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag4 tag, const IVect<dim>& IJK, BGK_Collider& collider) const
    {
        //std::cout<<"Je suis dans setup_collider Eq 4"<< std::endl;
        const real_t dt = Model.dt;
        const real_t dx = Model.dx;
        const real_t e2 = Model.e2;
        const real_t cs2 = 1.0 / e2 * SQR(dx / dt);

        LBMState lbmState;
        Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmState);
        const real_t M0 = Model.M0_PHI2(tag, lbmState);
        const RVect<dim> M1 = Model.M1_PHI2<dim>(tag, lbmState);
        const real_t M2 = Model.M2_PHI2(tag, lbmState);

        const real_t G2_st = Model.S_phase_change_phi2(lbmState); // Term proportional to lambda
        const real_t G2_dw = Model.S2_dw(lbmState); // Derivative of double-well

        const RVect<dim> G1_ct = Model.S_ct<dim>(lbmState);
        const RVect<dim> G2_ct = Model.S2_ct<dim>(lbmState);
        const RVect<dim> G3_ct = Model.S3_ct<dim>(lbmState);
        RVect<dim> G123_ct;
        G123_ct[IX] = (2.0 / 3.0) * G2_ct[IX] - (1.0 / 3.0) * G1_ct[IX] - (1.0 / 3.0) * G3_ct[IX];
        G123_ct[IY] = (2.0 / 3.0) * G2_ct[IY] - (1.0 / 3.0) * G1_ct[IY] - (1.0 / 3.0) * G3_ct[IY];
        G123_ct[IZ] = (2.0 / 3.0) * G2_ct[IZ] - (1.0 / 3.0) * G1_ct[IZ] - (1.0 / 3.0) * G3_ct[IZ];
        // compute collision rate
        collider.tau = Model.tau_PHI2(tag, lbmState);
        real_t staudx = 1 / ((collider.tau - 0.5) * dx / e2);
        // ipop = 0
        collider.f[0] = Base::get_f_val(tag, IJK, 0);
        collider.S0[0] = w[0] * dt * (G2_st + G2_dw + staudx * Base::compute_scal(0, G123_ct)); // MODIF ALAIN
        collider.feq[0] = M0 - (1 - w[0]) * M2 - 0.5 * collider.S0[0];
        // ipop > 0
        for (int ipop = 1; ipop < npop; ++ipop) {
            collider.f[ipop] = this->get_f_val(tag, IJK, ipop);
            collider.S0[ipop] = w[ipop] * dt * (G2_st + G2_dw + staudx * Base::compute_scal(ipop, G123_ct));
            collider.feq[ipop] = w[ipop] * (M2 + dx / dt * Base::compute_scal(ipop, M1) / cs2) - 0.5 * (collider.S0[ipop]);
        }
    } // end of setup_collider for phase field equation

    // =================== NAVIER-STOKES EQUATION ==========================
    //
    // 	VERSION AVEC COLLISION BGK

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag2 tag, const IVect<dim>& IJK, BGK_Collider& collider) const
    {
        //std::cout<<"Je suis dans setup_collider Eq 2"<< std::endl;
        // Paramètres pour la simulation
        const real_t dx = Model.dx;
        const real_t dt = Model.dt;
        const real_t cs2 = SQR(dx / dt) / Model.e2;

        // Stockage des anciennes grandeurs macroscopiques
        LBMState lbmState;
        Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmState);

        // Calcul du tau de collision
        collider.tau = Model.tau_NS(lbmState);

        // Calcul de l'équilibre sans terme source
        FState GAMMA;
        real_t scalUU = SQR(lbmState[IU]) + SQR(lbmState[IV]) + SQR(lbmState[IW]);
        for (int ipop = 0; ipop < npop; ++ipop) {
            real_t scalUC = dx / dt * Base::compute_scal(ipop, lbmState[IU], lbmState[IV], lbmState[IW]);
            GAMMA[ipop] = scalUC / cs2 + 0.5 * SQR(scalUC) / SQR(cs2) - 0.5 * scalUU / cs2;
            real_t feqbar = w[ipop] * (lbmState[IP] / (cs2 * lbmState[ID]) + GAMMA[ipop]);
            collider.feq[ipop] = feqbar;
            collider.f[ipop] = Base::get_f_val(tag, IJK, ipop);
        }

        // Calculs des termes forces pour le terme source

        //RVect<dim> ForceTS = Model.force_TS<dim>(lbmState);
        // AJOUTS ALAIN
        RVect<dim> ForceTS1 = Model.force_TS1<dim>(lbmState);
        RVect<dim> ForceTS2 = Model.force_TS2<dim>(lbmState); // AJOUTS ALAIN
        RVect<dim> ForceTS3 = Model.force_TS3<dim>(lbmState); // AJOUTS ALAIN
        // FIN AJOUTS ALAIN
        RVect<dim> ForceG = Model.force_G<dim>(lbmState);
        RVect<dim> ForceP = Model.force_P2<dim>(lbmState);
        RVect<dim> ForceV;
        RVect<dim> ForceTot;

        // Calcul de la force correctrice de Fakhari et al

        const real_t nu = Model.nu0 * Model.nu1 / (((1.0 - lbmState[IPHI]) * Model.nu1) + ((lbmState[IPHI]) * Model.nu0));
        real_t coeffV = -(nu * (Model.rho1 - Model.rho0)) / (cs2 * dt * collider.tau);

        ForceV[IX] = 0.0;
        ForceV[IY] = 0.0;
        ForceV[IZ] = 0.0;

        //for(int alpha=0; alpha<npop; ++alpha)
        //{
        //		ForceV[IX] += E[alpha][IX]*E[alpha][IX]*lbmState[IDPHIDX]*(collider.f[alpha]-collider.feq[alpha])+ E[alpha][IX]*E[alpha][IY]*lbmState[IDPHIDY]*(collider.f[alpha]-collider.feq[alpha]);
        //		ForceV[IY] += E[alpha][IY]*E[alpha][IX]*lbmState[IDPHIDX]*(collider.f[alpha]-collider.feq[alpha])+ E[alpha][IY]*E[alpha][IY]*lbmState[IDPHIDY]*(collider.f[alpha]-collider.feq[alpha]);
        //}
        // Modif force visqueuse : on remplace gradphi par gradrho
        for (int alpha = 0; alpha < npop; ++alpha) {
            ForceV[IX] += E[alpha][IX] * E[alpha][IX] * lbmState[IDRHODX] * (collider.f[alpha] - collider.feq[alpha]) + E[alpha][IX] * E[alpha][IY] * lbmState[IDRHODY] * (collider.f[alpha] - collider.feq[alpha]) + E[alpha][IX] * E[alpha][IZ] * lbmState[IDRHODZ] * (collider.f[alpha] - collider.feq[alpha]);
            ForceV[IY] += E[alpha][IY] * E[alpha][IX] * lbmState[IDRHODX] * (collider.f[alpha] - collider.feq[alpha]) + E[alpha][IY] * E[alpha][IY] * lbmState[IDRHODY] * (collider.f[alpha] - collider.feq[alpha]) + E[alpha][IY] * E[alpha][IZ] * lbmState[IDRHODZ] * (collider.f[alpha] - collider.feq[alpha]);
            ForceV[IZ] += E[alpha][IZ] * E[alpha][IX] * lbmState[IDRHODX] * (collider.f[alpha] - collider.feq[alpha]) + E[alpha][IZ] * E[alpha][IY] * lbmState[IDRHODY] * (collider.f[alpha] - collider.feq[alpha]) + E[alpha][IZ] * E[alpha][IZ] * lbmState[IDRHODZ] * (collider.f[alpha] - collider.feq[alpha]);
        }

        ForceV[IX] = coeffV * ForceV[IX];
        ForceV[IY] = coeffV * ForceV[IY];
        ForceV[IZ] = coeffV * ForceV[IZ];

        // Calcul et enregistrement du terme source total AJOUTS ALAIN TS2 et TS3
        ForceTot[IX] = ForceG[IX] + ForceP[IX] + ForceTS1[IX] + ForceTS2[IX] + ForceTS3[IX] + ForceV[IX];
        this->set_lbm_val(IJK, IFX, ForceTot[IX]);
        ForceTot[IY] = ForceG[IY] + ForceP[IY] + ForceTS1[IY] + ForceTS2[IY] + ForceTS3[IY] + ForceV[IY];
        this->set_lbm_val(IJK, IFY, ForceTot[IY]);
        ForceTot[IZ] = ForceG[IZ] + ForceP[IZ] + ForceTS1[IZ] + ForceTS2[IZ] + ForceTS3[IZ] + ForceV[IZ];
        this->set_lbm_val(IJK, IFZ, ForceTot[IZ]);

        // Rajout de la source dans l'équilibre
        for (int ipop = 0; ipop < npop; ++ipop) {
            collider.S0[ipop] = dt * w[ipop] / (lbmState[ID] * cs2) * Base::compute_scal(ipop, ForceTot[IX], ForceTot[IY], ForceTot[IZ]);
            collider.feq[ipop] = collider.feq[ipop] - 0.5 * (collider.S0[ipop]);
        }

    } // end of setup_collider for composition equation
    //
    //
    // =================== NAVIER-STOKES EQUATION MRT COLLISION ==========================
    //
    // VERSION AVEC MRT SETUP_COLLIDER FOR NS - Fakhari et. al. model (2017)
    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag2 tag, const IVect<dim>& IJK, MRT_Collider& collider) const
    {
        //std::cout<<"Je suis dans setup_collider Eq 2 MRT"<< std::endl;

        const real_t dx = Model.dx;
        const real_t dt = Model.dt;
        const real_t cs2 = SQR(dx / dt) / Model.e2;

        LBMState lbmState;
        Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmState);

        // compute collision rate
        //collider.tau = Model.tau_NS(lbmState);
        collider.tau = Model.tau_3phases_NS(lbmState);

        for (int i = 0; i < npop; i++) {
            if (Model.tauMatrixNS[i] != -1.0) {
                collider.S[i] = Model.tauMatrixNS[i];
            } else {
                collider.S[i] = 1 / collider.tau;
            };
        }

        FState GAMMA;
        // MODIFS ALAIN
        //RVect<dim> gradrho = Model.grad_rho<dim>(lbmState);
        RVect<dim> gradrho;
        gradrho[IX] = lbmState[IDRHODX];
        gradrho[IY] = lbmState[IDRHODY];
        gradrho[IZ] = lbmState[IDRHODZ];
        //RVect<dim> ForceTS = Model.force_TS<dim>(lbmState);
        RVect<dim> ForceTS1 = Model.force_TS1<dim>(lbmState); // AJOUTS ALAIN
        RVect<dim> ForceTS2 = Model.force_TS2<dim>(lbmState); // AJOUTS ALAIN
        RVect<dim> ForceTS3 = Model.force_TS3<dim>(lbmState); // AJOUTS ALAIN
        RVect<dim> ForceBck = Model.force_Beckermann<dim>(lbmState);
        MAYBE_UNUSED(ForceBck);

        //RVect<dim> ForceP = Model.force_P2<dim>(lbmState);
        RVect<dim> ForceP = Model.force_P_rhonum<dim>(lbmState); // MODIF ALAIN
        // FIN MODIFS ALAIN

        RVect<dim> ForceG = Model.force_G<dim>(lbmState);
        RVect<dim> ForceV;
        RVect<dim> ForceTot;

        real_t scalUU = SQR(lbmState[IU]) + SQR(lbmState[IV]) + SQR(lbmState[IW]);

        for (int ipop = 0; ipop < npop; ++ipop) {

            real_t scalUC = dx / dt * Base::compute_scal(ipop, lbmState[IU], lbmState[IV], lbmState[IW]);
            GAMMA[ipop] = scalUC / cs2 + 0.5 * SQR(scalUC) / SQR(cs2) - 0.5 * scalUU / cs2;
            real_t feqbar = w[ipop] * (lbmState[IP] / (cs2 * lbmState[ID]) + GAMMA[ipop]);

            collider.feq[ipop] = feqbar;
            collider.f[ipop] = Base::get_f_val(tag, IJK, ipop);
        }

        // Calcul de la source
        //const real_t nu = Model.nu0*Model.nu1/( ((1.0-lbmState[IPHI]) * Model.nu1) + ((lbmState[IPHI]) * Model.nu0));
        const real_t nu = lbmState[IPHI3] * Model.nu0 + lbmState[IPHI] * Model.nu1 + lbmState[IPHI2] * Model.nu2;
        real_t coeffV = -(nu) / (cs2 * dt);
        collider.Calc_Lf();
        FState MinvSMg_geq = collider.Prod_invMSMf;

        // FORCE VISQUEUSE
        ForceV[IX] = 0.0;
        ForceV[IY] = 0.0;
        ForceV[IZ] = 0.0;

        for (int beta = 0; beta < npop; ++beta) {
            ForceV[IX] += E[beta][IX] * E[beta][IX] * gradrho[IX] * MinvSMg_geq[beta] + E[beta][IX] * E[beta][IY] * gradrho[IY] * MinvSMg_geq[beta] + E[beta][IX] * E[beta][IZ] * gradrho[IZ] * MinvSMg_geq[beta];
            ForceV[IY] += E[beta][IY] * E[beta][IX] * gradrho[IX] * MinvSMg_geq[beta] + E[beta][IY] * E[beta][IY] * gradrho[IY] * MinvSMg_geq[beta] + E[beta][IY] * E[beta][IZ] * gradrho[IZ] * MinvSMg_geq[beta];
            ForceV[IZ] += E[beta][IZ] * E[beta][IX] * gradrho[IX] * MinvSMg_geq[beta] + E[beta][IZ] * E[beta][IY] * gradrho[IY] * MinvSMg_geq[beta] + E[beta][IZ] * E[beta][IZ] * gradrho[IZ] * MinvSMg_geq[beta];
        }

        ForceV[IX] = coeffV * ForceV[IX];
        ForceV[IY] = coeffV * ForceV[IY];
        ForceV[IZ] = coeffV * ForceV[IZ];

        // Calcul et enregistrement de la force totale AJOUTS ALAIN TS2 et TS3
        ForceTot[IX] = ForceG[IX] + ForceP[IX] + ForceTS1[IX] + ForceTS2[IX] + ForceTS3[IX] + ForceV[IX];
        this->set_lbm_val(IJK, IFX, ForceTot[IX]);
        ForceTot[IY] = ForceG[IY] + ForceP[IY] + ForceTS1[IY] + ForceTS2[IY] + ForceTS3[IY] + ForceV[IY];
        this->set_lbm_val(IJK, IFY, ForceTot[IY]);
        ForceTot[IZ] = ForceG[IZ] + ForceP[IZ] + ForceTS1[IZ] + ForceTS2[IZ] + ForceTS3[IZ] + ForceV[IZ];
        this->set_lbm_val(IJK, IFZ, ForceTot[IZ]);

        // Rajout de la source dans l'équilibre
        for (int ipop = 0; ipop < npop; ++ipop) {
            collider.S0[ipop] = dt * w[ipop] / (lbmState[ID] * cs2) * Base::compute_scal(ipop, ForceTot[IX], ForceTot[IY], ForceTot[IZ]);
            collider.feq[ipop] = collider.feq[ipop] - 0.5 * (collider.S0[ipop]);
        }
    }

    // ==================== COMPOSITION EQUATION ===========================

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag3 tag, const IVect<dim>& IJK, BGK_Collider& collider) const
    {
        //std::cout<<"Je suis dans setup_collider Eq 3"<< std::endl;
        const real_t dt = Model.dt;
        const real_t dx = Model.dx;
        const real_t e2 = Model.e2;
        const real_t cs2 = 1.0 / e2 * SQR(dx / dt);

        LBMState lbmState;
        Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmState);

        const real_t M0 = Model.M0_C(lbmState);
        const RVect<dim> M1 = Model.M1_C<dim>(lbmState);
        const real_t M2 = Model.M2_C(type, lbmState);
        const RVect<dim> S0 = Model.S0_C<dim>(type, lbmState);

        // compute collision rate
        // ATTENTION La ligne est ci-dessous est commentee!!!!!!
        collider.tau = Model.tau_C(type, lbmState);
        //collider.tau = Model.tau_C02(type, lbmState);
        //~ collider.tau = Model.tau_3phases_Compos(lbmState);

        // ipop = 0
        collider.f[0] = Base::get_f_val(tag, IJK, 0);
        collider.S0[0] = w[0] * dx * Base::compute_scal(0, S0);
        collider.feq[0] = M0 - (1 - w[0]) * M2 - 0.5 * collider.S0[0];

        // ipop > 0
        for (int ipop = 1; ipop < npop; ++ipop) {
            collider.f[ipop] = this->get_f_val(tag, IJK, ipop);
            collider.S0[ipop] = w[ipop] * dx * Base::compute_scal(ipop, S0);
            collider.feq[ipop] = w[ipop] * (M2 + Model.C_advec * dx / dt * Base::compute_scal(ipop, M1) / cs2) - 0.5 * collider.S0[ipop];
        }
    }

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag3 tag, const IVect<dim>& IJK, MRT_Collider& collider) const
    {

        const real_t dt = Model.dt;
        const real_t dx = Model.dx;
        const real_t e2 = Model.e2;
        const real_t cs2 = 1.0 / e2 * SQR(dx / dt);

        LBMState lbmState;
        Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmState);

        const real_t M0 = Model.M0_C(lbmState);
        const RVect<dim> M1 = Model.M1_C<dim>(lbmState);
        const real_t M2 = Model.M2_C(type, lbmState);
        const RVect<dim> S0 = Model.S0_C<dim>(type, lbmState);

        // compute collision rate
        //collider.tau = Model.tau_C(type, lbmState);
        collider.tau = Model.tau_C02(type, lbmState);

        for (int i = 0; i < npop; i++) {
            if (Model.tauMatrixComp[i] != -1.0) {
                collider.S[i] = Model.tauMatrixComp[i];
            } else {
                collider.S[i] = 1 / collider.tau;
            }
        }

        // ipop = 0
        collider.f[0] = Base::get_f_val(tag, IJK, 0);
        collider.S0[0] = w[0] * dx * Base::compute_scal(0, S0);
        collider.feq[0] = M0 - (1 - w[0]) * M2 - 0.5 * collider.S0[0];

        // ipop > 0
        for (int ipop = 1; ipop < npop; ++ipop) {
            collider.f[ipop] = this->get_f_val(tag, IJK, ipop);
            collider.S0[ipop] = w[ipop] * dx * Base::compute_scal(ipop, S0);
            collider.feq[ipop] = w[ipop] * (M2 + dx / dt * Base::compute_scal(ipop, M1) / cs2) - 0.5 * collider.S0[ipop];
        }
    }

    // =====================================================================
    //						MAKE BOUNDARY
    // =====================================================================

    KOKKOS_INLINE_FUNCTION
    void make_boundary(const IVect<dim>& IJK, int faceId) const
    {

        // phi boundary
        if (Base::params.boundary_types[BOUNDARY_EQUATION_1][faceId] == BC_ANTI_BOUNCE_BACK) {
            real_t boundary_value = Base::params.boundary_values[BOUNDARY_PHASE_FIELD][faceId];
            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_antibounceback(tagPHI, faceId, IJK, ipop, boundary_value);

        }

        else if (Base::params.boundary_types[BOUNDARY_EQUATION_1][faceId] == BC_ZERO_FLUX) {
            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_bounceback(tagPHI, faceId, IJK, ipop, 0.0);
        }
        // phi2 boundary AJOUT ALAIN
        // pour le moment les mêmes que Eq. phi
        if (Base::params.boundary_types[BOUNDARY_EQUATION_4][faceId] == BC_ANTI_BOUNCE_BACK) {
            real_t boundary_value = Base::params.boundary_values[BOUNDARY_PHASE_FIELD][faceId];
            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_antibounceback(tagPHI2, faceId, IJK, ipop, boundary_value);

        }

        else if (Base::params.boundary_types[BOUNDARY_EQUATION_4][faceId] == BC_ZERO_FLUX) {
            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_bounceback(tagPHI2, faceId, IJK, ipop, 0.0);
        }

        // NS boundaries
        if (Base::params.boundary_types[BOUNDARY_EQUATION_2][faceId] == BC_ANTI_BOUNCE_BACK) {
            real_t boundary_value = this->params.boundary_values[BOUNDARY_PRESSURE][faceId];
            for (int ipop = 0; ipop < npop; ++ipop) {
                this->compute_boundary_antibounceback(tagNS, faceId, IJK, ipop, boundary_value);
            }
        }

        else if (Base::params.boundary_types[BOUNDARY_EQUATION_2][faceId] == BC_ZERO_FLUX) {
            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_bounceback(tagNS, faceId, IJK, ipop, 0.0);

        }

        else if (Base::params.boundary_types[BOUNDARY_EQUATION_2][faceId] == BC_BOUNCEBACK) {
            const real_t dx = this->params.dx;
            const real_t dt = this->params.dt;
            const real_t cs2 = SQR(dx / dt) / Model.e2;
            real_t boundary_vx = this->params.boundary_values[BOUNDARY_VELOCITY_X][faceId];
            real_t boundary_vy = this->params.boundary_values[BOUNDARY_VELOCITY_Y][faceId];

            real_t value;
            for (int ipop = 0; ipop < npop; ++ipop) {
                value = dx / dt * this->compute_scal_Ebar(ipop, boundary_vx, boundary_vy) / cs2 * Model.rho0;
                this->compute_boundary_bounceback(tagNS, faceId, IJK, ipop, value);
            }

        } else if (Base::params.boundary_types[BOUNDARY_EQUATION_2][faceId] == BC_POISEUILLE) {
            const real_t dx = this->params.dx;
            const real_t dt = this->params.dt;
            const real_t isize = this->params.isize;
            const real_t jsize = this->params.jsize;
            const real_t cs2 = SQR(dx / dt) / Model.e2;
            //~ const real_t cs = SQRT(cs2);
            real_t scaling = (faceId == FACE_YMIN or faceId == FACE_YMAX) * 4 * IJK[IX] * (isize - IJK[IX]) / SQR(isize)
                + (faceId == FACE_XMIN or faceId == FACE_XMAX) * 4 * IJK[IY] * (jsize - IJK[IY]) / SQR(jsize);
            real_t boundary_vx = scaling * this->params.boundary_values[BOUNDARY_VELOCITY_X][faceId];
            real_t boundary_vy = scaling * this->params.boundary_values[BOUNDARY_VELOCITY_Y][faceId];

            real_t value;
            for (int ipop = 0; ipop < npop; ++ipop) {
                value = dx / dt * this->compute_scal_Ebar(ipop, boundary_vx, boundary_vy) / cs2 * Model.rho0;
                this->compute_boundary_bounceback(tagNS, faceId, IJK, ipop, value);
            }
        }

        // Composition boundaries
        if (Base::params.boundary_types[BOUNDARY_EQUATION_3][faceId] == BC_ANTI_BOUNCE_BACK) {
            real_t boundary_value = Base::params.boundary_values[BOUNDARY_CONCENTRATION][faceId];
            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_antibounceback(tagCOMP, faceId, IJK, ipop, boundary_value);

        }

        else if (Base::params.boundary_types[BOUNDARY_EQUATION_3][faceId] == BC_ZERO_FLUX) {
            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_bounceback(tagCOMP, faceId, IJK, ipop, 0.0);
        }
    };

    // =====================================================================
    //							INIT MACRO
    // =====================================================================

    KOKKOS_INLINE_FUNCTION
    void init_macro(IVect<dim> IJK, RANDOM_POOL::generator_type rand_gen) const
    {

        // get local coordinates
        real_t x = 0.0, y = 0.0, z = 0.0;
        this->get_coord3D(IJK, x, y, z);

        // get useful params
        real_t xphi = 0.0;
        real_t xphi2 = 0.0; // AJOUT ALAIN
        //~ real_t xphi3 = 0.0; // AJOUT ALAIN
        real_t xc = 0.0;

        real_t phi = 0.0;
        real_t phi2 = 0.0; // AJOUT ALAIN
        real_t phi3 = 0.0; // AJOUT ALAIN
        real_t rho = 0.0;
        real_t vx = 0.0;
        real_t vy = 0.0;
        real_t vz = 0.0;
        real_t p = 0.0;
        real_t c = 0.0;
        real_t mu = 0.0;
        //~ real_t cinit = 1.0;

        // Computation

        if (Model.initType == PHASE_FIELD_INIT_VERTICAL)
            xphi = x - Model.x0;

        else if (Model.initType == PHASE_FIELD_INIT_VERTICAL_3_PHASES)
        {
			// need x12>x01
            //~ xphi = abs(Model.x01-Model.x12)/2 - abs(x - (Model.x01+Model.x12)/2);
            if (Model.init_sharp)
            {
				xphi = Model.x01 - x;
	            xphi2 = x - Model.x12;
	            phi3 = (xphi>0);
	            phi2 = (xphi2>0);
	            phi=1-phi2-phi3;
			} else
			{
	            xphi = Model.x01 - x;
	            xphi2 = x - Model.x12;
	            phi3 = Model.phi0(xphi);
	            phi2 = Model.phi0(xphi2);
	            phi=1-phi2-phi3;
			}
		}
        else if (Model.initType == PHASE_FIELD_INIT_VERTICAL_3_PHASES_ERFC)
        {
			real_t time=Model.init_time;
			real_t x01 = Model.xi01 * SQRT(time);
			real_t x12 = Model.xi12 * SQRT(time);
            xphi =  x01 - x;
            xphi2 = x - x12;
            phi3 = Model.phi0(xphi);
            phi2 = Model.phi0(xphi2);
            phi=1-phi2-phi3;
			
			real_t A0 = Model.c0_inf;
			real_t B0 = ((Model.mu01 + Model.m0)-Model.c0_inf) / erfc(-Model.xi01/ (2 * SQRT(Model.D0)));
			
			real_t B1 = ((Model.mu01 + Model.m1)-(Model.mu12 + Model.m1)) / (erfc(Model.xi01/ (2 * SQRT(Model.D1))) - erfc(Model.xi12/ (2 * SQRT(Model.D1))));
			real_t A1 = (Model.mu01 + Model.m1) - B1 * erfc(Model.xi01 / (2 * SQRT(Model.D1)));
			
			real_t A2 = Model.c2_inf;
			real_t B2 = ((Model.mu12 + Model.m2)-Model.c2_inf) / erfc(Model.xi12/ (2 * SQRT(Model.D2)));
			
			real_t c0 = A0+ B0 * erfc(-x/ (2*SQRT(Model.D0 * time)));
			
			real_t c1 = A1+ B1 * erfc( x/ (2*SQRT(Model.D1 * time)));
			
			real_t c2 = A2+ B2 * erfc( x/ (2*SQRT(Model.D2 * time)));
			
			c = phi3 * c0 + phi * c1 + phi2 * c2;
			
			
		}
        else if (Model.initType == PHASE_FIELD_INIT_SPHERE)
        {
            xphi = (Model.r0 - sqrt(SQR(x - Model.x0) + SQR(y - Model.y0)));
		}
        else if (Model.initType == PHASE_FIELD_INIT_2SPHERE) {
            xphi = (Model.R - sqrt(SQR(x - Model.xC) + SQR(y - Model.yC)));
            xphi2 = (Model.R1 - sqrt(SQR(x - Model.xC1) + SQR(y - Model.yC1)));
            // real_t d = sqrt( SQR(x-Model.x0) + SQR(y-Model.y0));
            // bool ri= (d>((Model.r0+Model.r1)/2));
            // xphi = ri*(Model.r0 - d)-(1-ri)*(Model.r1 - d);
        }

        else if (Model.initType == PHASE_FIELD_INIT_SQUARE)
            xphi = FMIN(Model.r0 - sqrt(SQR(x - Model.x0)), Model.r1 - sqrt(SQR(y - Model.y0)));
        else if (Model.initType == PHASE_FIELD_INIT_MIXTURE)
        {
            
		}
        else if (Model.initType == PHASE_FIELD_INIT_TAYLOR) {
            real_t L = Model.wave_length;
            xphi = y - Model.ampl * sin(Model.n_wave * 2.0 * 3.14 / L * x + 3.14 / 2.0) - Model.height;
            xc = y - Model.ampl * sin(Model.n_wave * 2.0 * 3.14 / L * x + 3.14 / 2.0) - Model.height;
            c = Model.c0(xc);
        } else if (Model.initType == PHASE_FIELD_INIT_TAYLOR_3D) {
            real_t L = Model.wave_length;
            xphi = z - Model.ampl * sin(Model.n_wave * 2.0 * 3.14 / L * x + 3.14 / 2.0) - Model.height;
            phi = Model.phi0(xphi);
        } else if (Model.initType == TWO_PLANE) {
            real_t L = Model.wave_length;
            xphi = y - Model.ampl * sin(Model.n_wave * 2.0 * 3.14 / L * x + 3.14 / 2.0) - Model.height;
            xc = y - Model.ampl * sin(Model.n_wave * 2.0 * 3.14 / L * x + 3.14 / 2.0) - Model.height;
            vx = 0.5 * (Model.initVX_upper + Model.initVX_lower + (Model.initVX_upper - Model.initVX_lower) * tanh(Model.sign * 2.0 * xphi / Model.W));
            c = Model.c0(xc);
        } else if (Model.initType == BUBBLE_TAYLOR) {
            real_t L = Model.wave_length;
            real_t d1, d2;
            d1 = y - Model.ampl * sin(Model.n_wave * 2.0 * 3.14 / L * x + 3.14 / 2.0) - Model.height;
            d2 = Model.R - sqrt(SQR(x - Model.xC) + SQR(y - Model.yC));
            xphi = abs(d1) > abs(d2) ? d2 : d1;
            //xphi = d1;
            xc = xphi;
            c = Model.c0(xc);
        } else if (Model.initType == THREEPHASES_SPLASHING_DROPLETS) {
            //~ real_t L = Model.wave_length;MAYBE_UNUSED(L);
            real_t d1, d2, d3, d4;
            
            //~ real_t d0 = y - Model.height;
            d1 = Model.R - sqrt(SQR(x - Model.xC) + SQR(y - Model.yC));
            d2 = Model.R1 - sqrt(SQR(x - Model.xC1) + SQR(y - Model.yC1));
            d4 = Model.R2 - sqrt(SQR(x - Model.xC2) + SQR(y - Model.yC2));
            d3 = Model.height - y;
            if (y < Model.height + 10.0)
                phi = Model.phi0(d3);
            else if (y > Model.height)
                phi = Model.phi0(d1);
            xphi2 = d2;
            xc = d4;
            phi2 = Model.phi0(xphi2);
            //~ cinit = Model.phi0(xc);
        } else if (Model.initType == THREEPHASES_RISING_DROPLETS) {
            real_t L = Model.wave_length;
            real_t d0, d1, d2, d3;
            MAYBE_UNUSED(d0);
            MAYBE_UNUSED(L);
            d0 = y - Model.height;
            d1 = Model.R - sqrt(SQR(x - Model.xC) + SQR(y - Model.yC));
            d2 = Model.R1 - sqrt(SQR(x - Model.xC1) + SQR(y - Model.yC1));
            d3 = Model.height - y;
            if (y < Model.height + 10.0)
                phi = Model.phi0(d3);
            else if (y > Model.height)
                phi = Model.phi0(d1);
            xphi2 = d2;
            xc = xphi2;
            phi2 = Model.phi0(xphi2);
            phi -= phi2;
        } else if (Model.initType == THREEPHASES_RISING_DROPLETS_3D) {
            //~ real_t L = Model.wave_length;MAYBE_UNUSED(L);
			//~ real_t d0 = z - Model.height; MAYBE_UNUSED(d0);

            real_t  d1, d2, d3;
            
            
            d1 = Model.R - sqrt(SQR(x - Model.xC) + SQR(y - Model.yC) + SQR(z - Model.zC));
            d2 = Model.R1 - sqrt(SQR(x - Model.xC1) + SQR(y - Model.yC1) + SQR(z - Model.zC1));
            d3 = Model.height - z;
            if (z < Model.height + 3.0 * Model.dx)
                phi = Model.phi0(d3);
            else if (z > Model.height)
                phi = Model.phi0(d1);
            xphi2 = d2;
            xc = xphi2;
            phi2 = Model.phi0(xphi2);
            phi -= phi2;
        } else if (Model.initType == THREEPHASES_CONTAINER) {
            real_t xCmin, xCmax, yCmin, yCmax, Epaiss, Delta_x, d1, d2, d3, d4, d5;
            MAYBE_UNUSED(Epaiss);
            MAYBE_UNUSED(Delta_x);
            xCmin = Model.xmin_Container;
            xCmax = Model.xmax_Container;
            yCmin = Model.ymin_Container;
            yCmax = Model.ymax_Container;
            Epaiss = Model.Width_Container;
            Delta_x = 3. * Model.dx;
            //std::cout<<"xCmin :    "<<xCmin<<std::endl;
            //std::cout<<"xCmax :    "<<xCmax<<std::endl;
            //std::cout<<"yCmin :    "<<yCmin<<std::endl;
            //std::cout<<"yCmax :    "<<yCmax<<std::endl;
            //std::cout<<"Epaiss :    "<<Epaiss<<std::endl;
            d1 = x - xCmin;
            d2 = x - xCmax;
            d3 = y - yCmin;
            d4 = y - yCmax;
            d5 = Model.R1 - sqrt(SQR(x - Model.xC1) + SQR(y - Model.yC1));
            // Conditions sur phi
            phi = (Model.phi0(d1) - Model.phi0(d2)) * (Model.phi0(d3) - Model.phi0(d4));
            phi2 = Model.phi0(d5);
        } else if (Model.initType == THREEPHASES_RAYLEIGH_TAYLOR) {
            real_t L = Model.wave_length;
            real_t d0, d1;
            d0 = Model.height - y;
            d1 = Model.R - sqrt(SQR(x - Model.xC) + SQR(y - Model.yC));
            if (y < Model.height + 5 * Model.dx)
                phi = Model.phi0(d0);
            else if (y > Model.height)
                phi = Model.phi0(d1);
            xphi2 = Model.ampl * sin(Model.n_wave * 2.0 * 3.14 / L * x + 3.14 / 2.0) + Model.heightPhi2 - y;
            //xc    = Model.ampl*sin(Model.n_wave*2.0*3.14/L*x+3.14/2.0)+Model.heightPhi2 - y;
            phi2 = Model.phi0(xphi2);
            phi -= phi2;
        } else if (Model.initType == THREEPHASES_SPINODAL_DECOMPOSITION) {
            phi = Model.phi1moy + 0.01 * rand_gen.drand();
            phi2 = Model.phi2moy + 0.01 * rand_gen.drand();
            //RANDOM_POOL.free_state(rand_gen);
        } else if (Model.initType == THREEPHASES_SPREADING_LENS) {
            real_t a1 = sqrt(SQR(x - Model.xC) + SQR(y - Model.yC)) - Model.R;
            real_t b = y - Model.yC;
            real_t d1 = fmin(a1, b);
            real_t a2 = -sqrt(SQR(x - Model.xC) + SQR(y - Model.yC)) + Model.R;
            real_t d2 = fmax(a2, b);
            phi = Model.phi0(d1);
            phi2 = Model.phi2(d2);
        } else if (Model.initType == THREEPHASES_CAPSULE) {
            real_t d1, d2, xphi1, xphi2, xphi3;
            d1 = Model.R - sqrt(SQR(x - Model.xC) + SQR(y - Model.yC));
            xphi1 = FMIN(d1, Model.xC - x);
            d2 = Model.R1 - sqrt(SQR(x - Model.xC1) + SQR(y - Model.yC1));
            xphi2 = FMIN(d2, x - Model.xC1);
            phi = Model.phi0(xphi1) + Model.phi0(xphi2);
            xphi3 = FMIN(Model.width - sqrt(SQR(x - Model.x0)), Model.height - sqrt(SQR(y - Model.y0)));
            phi2 = Model.phi0(xphi3);
        } else if (Model.initType == PHASE_FIELD_INIT_SHRF_VORTEX) {
            real_t vx, vy, d1, d2;
            MAYBE_UNUSED(vx);
            MAYBE_UNUSED(vy);
            const real_t pi = M_PI;
            vx = -pi * Model.U0 * cos(pi * ((x / Model.L0) - 0.5)) * sin(pi * ((y / Model.L0) - 0.5));
            vy = pi * Model.U0 * sin(pi * ((x / Model.L0) - 0.5)) * cos(pi * ((y / Model.L0) - 0.5));
            d1 = Model.R - sqrt(SQR(x - Model.xC) + SQR(y - Model.yC));
            d2 = Model.R1 - sqrt(SQR(x - Model.xC1) + SQR(y - Model.yC1));
            phi = Model.phi0(d1);
            phi2 = Model.phi0(d2);
        }
        // Init phi3
        phi3 = 1.0 - phi - phi2;
        //c = phi*Model.c1_inf + phi3*Model.c0_inf + cinit*Model.c2_inf ;
        
        if (Model.initType == PHASE_FIELD_INIT_VERTICAL_3_PHASES)
        {	
			c = phi * Model.c1_inf + phi3 * Model.c0_inf + phi2 * Model.c2_inf;
			//~ real_t c1_int= (abs(x-Model.x01) * Model.c1_12 +  abs(x-Model.x12) * Model.c1_01)/abs(Model.x01-Model.x12);
			//~ c = phi * c1_int + phi3 * Model.c0_inf + phi2 * Model.c2_inf;
		}
		
        mu = Model.fun_mu(c, phi, phi2);
        // Save
        this->set_lbm_val(IJK, IPHI, phi);
        this->set_lbm_val(IJK, IPHI2, phi2);
        this->set_lbm_val(IJK, IPHI3, phi3);

        this->set_lbm_val(IJK, IC, c);
        this->set_lbm_val(IJK, IMU, mu);

        // init NS
        //rho = Model.interp_rho(phi,c);
        rho = phi3 * Model.rho0 + phi * Model.rho1 + phi2 * Model.rho2;
        this->set_lbm_val(IJK, IP, p);
        this->set_lbm_val(IJK, ID, rho);
        this->set_lbm_val(IJK, IU, vx);
        this->set_lbm_val(IJK, IV, vy);
        this->set_lbm_val(IJK, IW, vz);

        //std::cout<<"Je suis dans init_macro"<< std::endl;

    } // end init macro

    // =====================================================================
    //							UPDATE MACRO
    // =====================================================================

    KOKKOS_INLINE_FUNCTION
    void update_macro(IVect<dim> IJK) const
    {
        // get useful params
        const real_t dx = this->params.dx;
        const real_t dt = this->params.dt;
        const real_t cs2 = SQR(dx / dt) / Model.e2;

        // compute moments of distribution equations
        real_t moment_phi = 0.0;
        real_t moment_phi2 = 0.0; // AJOUT ALAIN
        real_t phi3 = 0.0; // AJOUT ALAIN
        real_t moment_P = 0.0;
        real_t moment_VX = 0.0;
        real_t moment_VY = 0.0;
        real_t moment_VZ = 0.0;
        real_t moment_c = 0.0;
        for (int ipop = 0; ipop < npop; ++ipop) {
            moment_phi += Base::get_f_val(tagPHI, IJK, ipop);
            moment_phi2 += Base::get_f_val(tagPHI2, IJK, ipop); // AJOUT ALAIN
            moment_P += Base::get_f_val(tagNS, IJK, ipop);
            moment_VX += Base::get_f_val(tagNS, IJK, ipop) * E[ipop][IX];
            moment_VY += Base::get_f_val(tagNS, IJK, ipop) * E[ipop][IY];
            moment_VZ += Base::get_f_val(tagNS, IJK, ipop) * E[ipop][IZ];
            moment_c += Base::get_f_val(tagCOMP, IJK, ipop);
        }

        // store old values of macro fields
        LBMState lbmStatePrev;
        Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmStatePrev);

        const real_t rhoprev = lbmStatePrev[ID];
        MAYBE_UNUSED(rhoprev);
        const real_t ForceNSX = lbmStatePrev[IFX];
        const real_t ForceNSY = lbmStatePrev[IFY];
        const real_t ForceNSZ = lbmStatePrev[IFZ];

        // get source term
        const real_t G_st = Model.S_phase_change_phi1( lbmStatePrev);
        const real_t G_dw = Model.S_dw(tagPHI, lbmStatePrev);

        //
        const real_t G2_st = Model.S_phase_change_phi2(lbmStatePrev); // AJOUT ALAIN
        const real_t G2_dw = Model.S2_dw(lbmStatePrev); // AJOUT ALAIN

        // compute new macro fields
        const real_t phi = moment_phi + 0.5 * dt * (G_st + G_dw);
        // UPDATE PHI2 and COMPUTE PHI3
        const real_t phi2 = moment_phi2 + 0.5 * dt * (G2_st + G2_dw); // AJOUT ALAIN
        phi3 = 1.0 - phi - phi2; // AJOUT ALAIN

        const real_t c = moment_c;
        //const real_t mu  = Model.mu(c, phi);
        const real_t mu = Model.fun_mu(c, phi, phi2);
        // MODIF ALAIN rho mis en commentaires et remplacé
        //const real_t rho = Model.interp_rho(phi,c);
        //const real_t rho =  phi3*Model.rho0 + phi*Model.rho1 + phi2*Model.rho2 ;          // AJOUT ALAIN
        const real_t rho = phi3 * Model.rho0c(c) + phi * Model.rho1c(c) + phi2 * Model.rho2c(c); // AJOUT ALAIN

        // compute NS macro vars
        //const real_t P = moment_P*rhoprev*cs2;
        //const real_t VX = moment_VX + 0.5 * dt * ForceNSX / rhoprev;
        //const real_t VY = moment_VY + 0.5 * dt * ForceNSY / rhoprev;
        const real_t P = moment_P * rho * cs2; // Mise A Jour pression avec rho calculé au-dessus
        const real_t VX = moment_VX + 0.5 * dt * ForceNSX / rho;
        const real_t VY = moment_VY + 0.5 * dt * ForceNSY / rho;
        const real_t VZ = moment_VZ + 0.5 * dt * ForceNSZ / rho;
        //
        // =====================================================================================
        // Pour le cas test de la "capsule" pour laisser diffuser l'interface
        // (due à l'initialisation) avant de calculer le NS
        //real_t VX, VY, P;
        //if (Model.time > 100.*Model.dt) {
        //P  = moment_P*rho*cs2; // Mise A Jour pression avec rho calculé au-dessus
        //VX = moment_VX + 0.5 * dt * ForceNSX / rho;
        //VY = moment_VY + 0.5 * dt * ForceNSY / rho;
        //}
        //else {
        //VX  = 0.0;
        //VY  = 0.0;
        //P   = 0.0;
        //}
        //
        // =====================================================================================
        // Pour le test du vortex si on travaille salement sans découpler le NS
        //real_t VX, VY, x, y;
        //this->get_coordinates(IJK, x, y);
        //const real_t pi = M_PI;
        //VX = 0.0;
        //VY = 0.0;
        //VX = - pi*Model.U0 * cos(pi*((x/Model.L0)-0.5)) * sin(pi*((y/Model.L0)-0.5));
        //VY =   pi*Model.U0 * sin(pi*((x/Model.L0)-0.5)) * cos(pi*((y/Model.L0)-0.5));
        //
        // =====================================================================================
        //
        // update macro fields
        this->set_lbm_val(IJK, IPHI, phi);
        this->set_lbm_val(IJK, IPHI2, phi2); // AJOUT ALAIN
        this->set_lbm_val(IJK, IPHI3, phi3); // AJOUT ALAIN

        this->set_lbm_val(IJK, ID, rho);
        this->set_lbm_val(IJK, IP, P);
        this->set_lbm_val(IJK, IU, VX);
        this->set_lbm_val(IJK, IV, VY);
        this->set_lbm_val(IJK, IW, VZ);
        this->set_lbm_val(IJK, IC, c);
        this->set_lbm_val(IJK, IMU, mu);
        //std::cout<<"Je suis dans update_macro"<< std::endl;

    } // end update macro

    // =====================================================================
    //							UPDATE MACRO GRAD
    // =====================================================================

    KOKKOS_INLINE_FUNCTION
    void update_macro_grad(IVect<dim> IJK) const
    {
        RVect<dim> gradPhi;
        this->compute_gradient(gradPhi, IJK, IPHI, BOUNDARY_EQUATION_1);
        this->set_lbm_val(IJK, IDPHIDX, gradPhi[IX]);
        this->set_lbm_val(IJK, IDPHIDY, gradPhi[IY]);
        this->set_lbm_val(IJK, IDPHIDZ, gradPhi[IZ]);
        // AJOUT ALAIN Attention BOUNDARY_EQUATION_1 en argument
        RVect<dim> gradPhi2;
        this->compute_gradient(gradPhi2, IJK, IPHI2, BOUNDARY_EQUATION_1);
        this->set_lbm_val(IJK, IDPHI2DX, gradPhi2[IX]);
        this->set_lbm_val(IJK, IDPHI2DY, gradPhi2[IY]);
        this->set_lbm_val(IJK, IDPHI2DZ, gradPhi2[IZ]);
        RVect<dim> gradPhi3;
        this->compute_gradient(gradPhi3, IJK, IPHI3, BOUNDARY_EQUATION_1);
        this->set_lbm_val(IJK, IDPHI3DX, gradPhi3[IX]);
        this->set_lbm_val(IJK, IDPHI3DY, gradPhi3[IY]);
        this->set_lbm_val(IJK, IDPHI3DZ, gradPhi3[IZ]);
        RVect<dim> gradRho;
        this->compute_gradient(gradRho, IJK, ID, BOUNDARY_EQUATION_2);
        this->set_lbm_val(IJK, IDRHODX, gradRho[IX]);
        this->set_lbm_val(IJK, IDRHODY, gradRho[IY]);
        this->set_lbm_val(IJK, IDRHODZ, gradRho[IZ]);
        // FIN AJOUT

        real_t laplaPhi = Base::compute_laplacian(IJK, IPHI, BOUNDARY_EQUATION_1);
        Base::set_lbm_val(IJK, ILAPLAPHI, laplaPhi);
        // AJOUT ALAIN (ATTENTION EQUATION_1 en ARGUMENT)
        real_t laplaPhi2 = Base::compute_laplacian(IJK, IPHI2, BOUNDARY_EQUATION_1);
        Base::set_lbm_val(IJK, ILAPLAPHI2, laplaPhi2);
        real_t laplaPhi3 = Base::compute_laplacian(IJK, IPHI3, BOUNDARY_EQUATION_1);
        Base::set_lbm_val(IJK, ILAPLAPHI3, laplaPhi3);
        // FIN AJOUT ALAIN

        RVect<dim> gradC;
        this->compute_gradient(gradC, IJK, IC, BOUNDARY_EQUATION_3);
        this->set_lbm_val(IJK, IDCDX, gradC[IX]);
        this->set_lbm_val(IJK, IDCDY, gradC[IY]);
        this->set_lbm_val(IJK, IDCDZ, gradC[IZ]);
        //std::cout<<"Je suis dans update_macro_grad"<< std::endl;
    }

}; // end class

} // end namespace
#endif
