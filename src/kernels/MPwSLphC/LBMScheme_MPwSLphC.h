#ifndef LBMSCHEME_MPWSLPHC_H_
#define LBMSCHEME_MPWSLPHC_H_

#include <limits> // for std::numeric_limits

#ifdef __CUDA_ARCH__
#include <math_constants.h>
#endif // __CUDA_ARCH__

#include "FieldManager.h"
#include "LBMParams.h"
#include "LBM_Base_Functor.h"

#include "Models_MPwSLphC.h"

namespace PBM_MPwSLphC {
template <int dim, int npop, typename modelType>
struct LBMScheme : public LBMSchemeBase<dim, npop> {
  using Base = LBMSchemeBase<dim, npop>;
  using LBMState = typename Kokkos::Array<real_t, COMPONENT_SIZE>;
  using id2index_t = Kokkos::Array<int, COMPONENT_SIZE>;
  using LBMArray = typename LBMBaseFunctor<dim, npop>::LBMArray;
  using FArray = typename LBMBaseFunctor<dim, npop>::FArray;
  using LBM_Weights = typename LBMBaseFunctor<dim, npop>::LBM_Weights;
  using LBM_speeds = typename LBMBaseFunctor<dim, npop>::LBM_speeds;
  using LBM_speeds_opposite =
      typename LBMBaseFunctor<dim, npop>::LBM_speeds_opposite;
  using FState = typename Kokkos::Array<real_t, npop>;

  static constexpr real_t NORMAL_EPSILON = 1.0e-16;

  using BGK_Collider              = BGKCollider<dim, npop>;
  using MRT_Collider              = MRTCollider<dim, npop>;
  using TRT_Collider              = TRTCollider<dim, npop>;
  using BGK_Collider_Time_Factor  = BGKColliderTimeFactor<dim,npop>;

  ModelParams Model;
  modelType type;
  LBM_speeds E;
  LBM_speeds_opposite Ebar;
  LBM_Weights w;
  EquationTag1 tagPHI;
  EquationTag2 tagNS;
  EquationTag3 tagCOMP;
  EquationTag4 tagPSI;
  
  LBMScheme()
      : LBMSchemeBase<dim, npop>(), E(LBMBaseFunctor<dim, npop>::E),
        Ebar(LBMBaseFunctor<dim, npop>::Ebar),
        w(LBMBaseFunctor<dim, npop>::w){};

  LBMScheme(ConfigMap configMap, LBMParams params, LBMArray &lbm_data)
      : LBMSchemeBase<dim, npop>(params, lbm_data), Model(configMap, params),
        E(LBMBaseFunctor<dim, npop>::E), Ebar(LBMBaseFunctor<dim, npop>::Ebar),
        w(LBMBaseFunctor<dim, npop>::w){};

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
  //                        SETUP COLLIDER
  // =====================================================================

  // ==================== PHASE FIELD EQUATION PHI========================

  // ====================          BGK         ===========================
  KOKKOS_INLINE_FUNCTION
  void setup_collider(EquationTag1 tag, const IVect<dim> &IJK,
                      BGK_Collider &collider) const {

    const real_t dt = Model.dt;
    const real_t dx = Model.dx;
    const real_t e2 = Model.e2;
    const real_t c = dx / dt;
    // const real_t cs2 = SQR(c)/Model.e2;
    const real_t c_cs2 = e2 / c; // Ratio c/cs2
	const int CH = Model.cahn_hilliard ;
    LBMState lbmState;
    Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmState);

    const real_t M0     = Model.M0_PHI(tag, lbmState);
    const RVect<dim> M1 = Model.M1_PHI<dim>(tag, lbmState);
    const real_t M2_AC  = Model.M2_PHI(tag, lbmState);
    //const real_t M2_CH  = Model.M2_MU_PHI(tag, lbmState);
    const real_t M2     = M2_AC ;

    const real_t G_st = Model.St_phi(tag, lbmState);
    const real_t G_dw = Model.S_dw(tag, lbmState);
    
    const RVect<dim> G_ct = Model.S_ct<dim>(lbmState);
    //const RVect<dim> G_ct = Model.S_ct_psi<dim>(lbmState);
	const RVect<dim> LagrMult = Model.Lagrange_mult<dim>(lbmState);
	
    // compute collision rate
    //collider.tau = Model.tau_PHI(tag, lbmState);
    collider.tau = Model.tau_PHI_sol(tag, lbmState);
    real_t staudx = e2 / ((collider.tau - 0.5) * dx);
    // ipop = 0
    collider.f[0] = Base::get_f_val(tag, IJK, 0);
    collider.S0[0] =
        w[0] * dt * (G_st + G_dw + staudx * Base::compute_scal(0, G_ct));
    collider.feq[0] = M0 - (1 - w[0]) * M2 - 0.5 * collider.S0[0];

    // ipop > 0
    for (int ipop = 1; ipop < npop; ++ipop) {
      collider.f[ipop] = this->get_f_val(tag, IJK, ipop);
      collider.S0[ipop] = w[ipop] * dt *
			(G_st + G_dw + staudx * Base::compute_scal(ipop, G_ct)); //// - Base::compute_scal(ipop, LagrMult)
      collider.feq[ipop] =
          w[ipop] * (M2 + c_cs2 * Base::compute_scal(ipop, M1)) - 0.5 * (collider.S0[ipop])
					- w[ipop] * Base::compute_scal(ipop, LagrMult) * c_cs2 ;
    }
  } // end of setup_collider for phase field equation
  
  // ====================          MRT         ===========================

  KOKKOS_INLINE_FUNCTION
  void setup_collider(EquationTag1 tag, const IVect<dim> &IJK,
                      MRT_Collider &collider) const {

    // NOT IMPLEMENTED
    const real_t dt = Model.dt;
    const real_t dx = Model.dx;
    const real_t e2 = Model.e2;
    const real_t c = dx / dt;
    // const real_t cs2 = SQR(c)/Model.e2;
    const real_t c_cs2 = e2 / c; // Ratio c/cs2

    LBMState lbmState;
    Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmState);

	// compute collision rate
    collider.tau = Model.tau_PHI(tag,lbmState);
    const real_t tauInv = 1 / collider.tau;

    // Remplissage de la matrice S
    for (int i = 0; i < npop; i++) {
      if (Model.tauMatrixAC[i] != -1.0) {
        collider.S[i] = Model.tauMatrixAC[i];
      } else {
        collider.S[i] = tauInv;
      }
    }
    const real_t M0 = Model.M0_PHI(tag, lbmState);
    const RVect<dim> M1 = Model.M1_PHI<dim>(tag, lbmState);
    const real_t M2 = Model.M2_PHI(tag, lbmState);

    const real_t G_st = Model.S_st(tag, lbmState);
    const real_t G_dw = Model.S_dw(tag, lbmState);
    const RVect<dim> G_ct = Model.S_ct<dim>(lbmState);

    real_t staudx = e2 / ((collider.tau - 0.5) * dx);
    // ipop = 0
    collider.f[0] = Base::get_f_val(tag, IJK, 0);
    collider.S0[0] =
        w[0] * dt * (G_st + G_dw + staudx * Base::compute_scal(0, G_ct));
    collider.feq[0] = M0 - (1 - w[0]) * M2 - 0.5 * collider.S0[0];

    // ipop > 0
    for (int ipop = 1; ipop < npop; ++ipop) {
      collider.f[ipop]   = this->get_f_val(tag, IJK, ipop);
      collider.S0[ipop]  = w[ipop] * dt * (G_st + G_dw + staudx * Base::compute_scal(ipop, G_ct));
	  collider.feq[ipop] = w[ipop] * (M2 + c_cs2 * Base::compute_scal(ipop, M1)) - 0.5 * (collider.S0[ipop]);
    }
    
  } // end of setup_collider for phase field equation
  //
  // ====================          TRT         ===========================
  KOKKOS_INLINE_FUNCTION
  void setup_collider(EquationTag1 tag, const IVect<dim> &IJK,
                      TRTCollider<dim, npop> &collider) const {



    const real_t dt = Model.dt;
    const real_t dx = Model.dx;
    const real_t e2 = Model.e2;
    const real_t c = dx/dt;
    //const real_t cs2 = SQR(c)/Model.e2;
    const real_t c_cs2 = e2/c; // Ratio c/cs2

    LBMState lbmState;
    // Base::template setupLBMState2<LBMState,COMPONENT_SIZE>(IJK, lbmState);
    this->setupLBMState(IJK, lbmState);

    const real_t M0 = Model.M0_PHI(tag, lbmState);
    const RVect<dim> M1 = Model.M1_PHI<dim>(tag, lbmState);
    const real_t M2 = Model.M2_PHI(tag, lbmState);

    const real_t G_st = Model.S_st(tag, lbmState);
    const real_t G_dw = Model.S_dw(tag, lbmState);
    const RVect<dim> G_ct = Model.S_ct<dim>(lbmState);

    // compute collision rate
    collider.tauA = Model.tau_PHI (tag, lbmState);
	// Verif BGK
    //collider.tauS = collider.tauA ;
    collider.tauS = 0.5+this->params.lambdaTRT1/(collider.tauA-0.5);
    
    real_t staudx = 1 / ((collider.tauA - 0.5) * dx / Model.e2);
    // ipop = 0
    collider.f[0]   = Base::get_f_val(tag, IJK, 0);
    collider.S0[0]  = w[0] * dt * (G_st + G_dw + staudx * Base::compute_scal(0, G_ct));
    collider.feq[0] = M0 - (1-w[0])*M2 - 0.5 * collider.S0[0];

    // ipop > 0
    for (int ipop = 1; ipop < npop; ++ipop) {
      collider.f[ipop] = this->get_f_val(tag, IJK, ipop);
      collider.S0[ipop] = w[ipop] * dt * (G_st + G_dw + staudx * Base::compute_scal(ipop, G_ct));
      collider.feq[ipop] = w[ipop] * (M2 + c_cs2 * Base::compute_scal(ipop, M1)) - 0.5 * (collider.S0[ipop]);    
    }
    
    // ipop = 0
//    collider.f[0]   = Base::get_f_val(tag, IJK, 0);
//    collider.S0[0]  = w[0] * dt * (G_st + G_dw);
//    collider.feq[0] = M0 - (1-w[0])*M2 - 0.5 * collider.S0[0];

    // ipop > 0
//    for (int ipop = 1; ipop < npop; ++ipop) {
//      collider.f[ipop] = this->get_f_val(tag, IJK, ipop);
//      collider.S0[ipop] = w[ipop] * dt * (G_st + G_dw);
//      collider.feq[ipop] = w[ipop] * (M2 + c_cs2 * (Base::compute_scal(ipop, M1)+ Base::compute_scal(ipop, G_ct)))
//									- 0.5 * (collider.S0[ipop]);    
//    }
  } // end of setup_collider for composition equation
 
  // ================================== NAVIER-STOKES EQUATION ===============================
  //
  //
  // ==================================    NS BGK COLLISION    ===============================
  // SETUP_COLLIDER FOR NS - Fakhari et. al. model (2017)
  //
  //
  KOKKOS_INLINE_FUNCTION
  void setup_collider(EquationTag2 tag, const IVect<dim> &IJK,
                      BGK_Collider &collider) const {

    // Paramètres pour la simulation
    const real_t dx = Model.dx;
    const real_t dt = Model.dt;
    const real_t c = dx / dt;
    const real_t cs2 = SQR(c) / Model.e2;

    // Stockage des anciennes grandeurs macroscopiques
    LBMState lbmState;
    Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmState);

    // Calcul du tau de collision
    collider.tau = Model.tau_NS(lbmState);

    // Calcul de l'équilibre sans terme source
    FState GAMMA;

    if (dim == 2) {
      real_t scalUU = SQR(lbmState[IU]) + SQR(lbmState[IV]);
      for (int ipop = 0; ipop < npop; ++ipop) {
        real_t scalUC =
            c * Base::compute_scal(ipop, lbmState[IU], lbmState[IV]);
        GAMMA[ipop] =
            scalUC / cs2 + 0.5 * SQR(scalUC) / SQR(cs2) - 0.5 * scalUU / cs2;
        real_t feqbar =
            w[ipop] * (lbmState[IP] / (cs2 * lbmState[ID]) + GAMMA[ipop]);
        collider.feq[ipop] = feqbar;
        collider.f[ipop] = Base::get_f_val(tag, IJK, ipop);
      }

      // Calculs des termes forces pour le terme source

      // RVect<dim> gradrho = Model.grad_rho<dim>(lbmState);
      RVect<dim> ForceTS = Model.force_TS<dim>(lbmState);

      RVect<dim> ForceG = Model.force_G<dim>(lbmState);
      RVect<dim> ForceP = Model.force_P_meth1<dim>(lbmState);
      RVect<dim> ForceV;
      RVect<dim> ForceTot;

      // Calcul de la force correctrice de Fakhari et al
      const real_t nu = Model.nu0 * Model.nu1 /
                        (((1.0 - lbmState[IPHI]) * Model.nu1) +
                         ((lbmState[IPHI]) * Model.nu0));
      real_t coeffV = - nu / (cs2 * dt * collider.tau);

      ForceV[IX] = 0.0;
      ForceV[IY] = 0.0;

      for (int alpha = 0; alpha < npop; ++alpha) {
        ForceV[IX] += E[alpha][IX] * E[alpha][IX] * lbmState[IDRHODX] *
                          (collider.f[alpha] - collider.feq[alpha]) +
                      E[alpha][IX] * E[alpha][IY] * lbmState[IDRHODY] *
                          (collider.f[alpha] - collider.feq[alpha]);
        ForceV[IY] += E[alpha][IY] * E[alpha][IX] * lbmState[IDRHODX] *
                          (collider.f[alpha] - collider.feq[alpha]) +
                      E[alpha][IY] * E[alpha][IY] * lbmState[IDRHODY] *
                          (collider.f[alpha] - collider.feq[alpha]);
      }

      ForceV[IX] = coeffV * ForceV[IX];
      ForceV[IY] = coeffV * ForceV[IY];

      // Calcul et enregistrement du terme source total
      ForceTot[IX] = ForceG[IX] + ForceP[IX] + ForceTS[IX] + ForceV[IX];
      ForceTot[IY] = ForceG[IY] + ForceP[IY] + ForceTS[IY] + ForceV[IY];

      this->set_lbm_val(IJK, IFX, ForceTot[IX]);
      this->set_lbm_val(IJK, IFY, ForceTot[IY]);

      // Rajout de la source dans l'équilibre
      for (int ipop = 0; ipop < npop; ++ipop) {
        collider.S0[ipop] =
            dx * w[ipop] / (lbmState[ID] * cs2) *
            Base::compute_scal(ipop, ForceTot[IX], ForceTot[IY]);
        collider.feq[ipop] = collider.feq[ipop] - 0.5 * (collider.S0[ipop]);
      }
    }

    if (dim == 3) {
      real_t scalUU = SQR(lbmState[IU]) + SQR(lbmState[IV]) + SQR(lbmState[IW]);
      for (int ipop = 0; ipop < npop; ++ipop) {
        real_t scalUC = c * Base::compute_scal(ipop, lbmState[IU], lbmState[IV],
                                               lbmState[IW]);
        GAMMA[ipop] =
            scalUC / cs2 + 0.5 * SQR(scalUC) / SQR(cs2) - 0.5 * scalUU / cs2;
        real_t feqbar =
            w[ipop] * (lbmState[IP] / (cs2 * lbmState[ID]) + GAMMA[ipop]);
        collider.feq[ipop] = feqbar;
        collider.f[ipop] = Base::get_f_val(tag, IJK, ipop);
      }

      // Calculs des termes forces pour le terme source

      // RVect<dim> gradrho = Model.grad_rho<dim>(lbmState);
      RVect<dim> ForceTS = Model.force_TS<dim>(lbmState);

      RVect<dim> ForceG = Model.force_G<dim>(lbmState);
      RVect<dim> ForceP = Model.force_P_meth1<dim>(lbmState);
      RVect<dim> ForceV;
      RVect<dim> ForceTot;

      // Calcul de la force correctrice de Fakhari et al
      const real_t nu = Model.nu0 * Model.nu1 /
                        (((1.0 - lbmState[IPHI]) * Model.nu1) +
                         ((lbmState[IPHI]) * Model.nu0));
      real_t coeffV =
          - nu / (cs2 * dt * collider.tau);

      ForceV[IX] = 0.0;
      ForceV[IY] = 0.0;
      ForceV[IZ] = 0.0;

      for (int alpha = 0; alpha < npop; ++alpha) {
        ForceV[IX] += E[alpha][IX] * E[alpha][IX] * lbmState[IDRHODX] *
                          (collider.f[alpha] - collider.feq[alpha]) +
                      E[alpha][IX] * E[alpha][IY] * lbmState[IDRHODY] *
                          (collider.f[alpha] - collider.feq[alpha]) +
                      E[alpha][IX] * E[alpha][IZ] * lbmState[IDRHODZ] *
                          (collider.f[alpha] - collider.feq[alpha]);
        ForceV[IY] += E[alpha][IY] * E[alpha][IX] * lbmState[IDRHODX] *
                          (collider.f[alpha] - collider.feq[alpha]) +
                      E[alpha][IY] * E[alpha][IY] * lbmState[IDRHODY] *
                          (collider.f[alpha] - collider.feq[alpha]) +
                      E[alpha][IY] * E[alpha][IZ] * lbmState[IDRHODZ] *
                          (collider.f[alpha] - collider.feq[alpha]);
        ForceV[IZ] += E[alpha][IZ] * E[alpha][IX] * lbmState[IDRHODX] *
                          (collider.f[alpha] - collider.feq[alpha]) +
                      E[alpha][IZ] * E[alpha][IY] * lbmState[IDRHODY] *
                          (collider.f[alpha] - collider.feq[alpha]) +
                      E[alpha][IZ] * E[alpha][IZ] * lbmState[IDRHODZ] *
                          (collider.f[alpha] - collider.feq[alpha]);
      }

      ForceV[IX] = coeffV * ForceV[IX];
      ForceV[IY] = coeffV * ForceV[IY];
      ForceV[IZ] = coeffV * ForceV[IZ];

      // Calcul et enregistrement du terme source total
      ForceTot[IX] = ForceG[IX] + ForceP[IX] + ForceTS[IX] + ForceV[IX];
      ForceTot[IY] = ForceG[IY] + ForceP[IY] + ForceTS[IY] + ForceV[IY];
      ForceTot[IZ] = ForceG[IZ] + ForceP[IZ] + ForceTS[IZ] + ForceV[IZ];

      this->set_lbm_val(IJK, IFX, ForceTot[IX]);
      this->set_lbm_val(IJK, IFY, ForceTot[IY]);
      this->set_lbm_val(IJK, IFZ, ForceTot[IZ]);

      // Rajout de la source dans l'équilibre
      for (int ipop = 0; ipop < npop; ++ipop) {
        collider.S0[ipop] =
            dt * w[ipop] / (lbmState[ID] * cs2) *
            Base::compute_scal(ipop, ForceTot[IX], ForceTot[IY], ForceTot[IZ]);
        collider.feq[ipop] = collider.feq[ipop] - 0.5 * (collider.S0[ipop]);
      }
    }

  } // end of setup_collider for composition equation
  //
  //
  // ================================== NS MRT COLLISION =====================================
  // SETUP_COLLIDER FOR NS - Fakhari et. al. model (2017)
  //
  //
  KOKKOS_INLINE_FUNCTION
  void setup_collider(EquationTag2 tag, const IVect<dim> &IJK,
                      MRT_Collider &collider) const {

    // Setup matrices MRT
    //collider.setMandMinv(Base::M, Base::Minv);

    // Paramètres pour la simulation
    const real_t dx = Model.dx;
    const real_t dt = Model.dt;
    const real_t c = dx / dt;
    const real_t cs2 = SQR(c) / Model.e2;
    const real_t cs2Inv = Model.e2 / SQR(c);

    // Stockage des anciennes grandeurs macroscopiques
    LBMState lbmState;
    Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmState);

    // compute collision rate
    //collider.tau = Model.tau_NS(lbmState);
    collider.tau = Model.tau_NS_harm12S(lbmState) ;
    //collider.tau = Model.tau_NS_harm_eta(lbmState);
    const real_t tauInv = 1 / collider.tau;

    // Remplissage de la matrice S
    for (int i = 0; i < npop; i++) {
      if (Model.tauMatrixNS[i] != -1.0) {
        collider.S[i] = Model.tauMatrixNS[i];
      } else {
        collider.S[i] = tauInv;
      }
    }

    // Calcul de l'équilibre sans terme source
    FState GAMMA;
	if (dim == 2) {

		real_t scalUU = SQR(lbmState[IU]) + SQR(lbmState[IV]) ;

		for (int ipop = 0; ipop < npop; ++ipop) {
			real_t scalUC = c * Base::compute_scal(ipop, lbmState[IU], lbmState[IV]);
			GAMMA[ipop]   = scalUC * cs2Inv + 0.5 * SQR(scalUC) * SQR(cs2Inv) - 0.5 * scalUU * cs2Inv;
			real_t feqbar = w[ipop] * (lbmState[IP] / (cs2 * lbmState[ID]) + GAMMA[ipop]);
			collider.feq[ipop] = feqbar;
			collider.f[ipop] = Base::get_f_val(tag, IJK, ipop);
		}

		// Calculs des termes forces pour le terme source
		// Surface tension
		RVect<dim> ForceTS_2liq = Model.force_TS<dim>(lbmState);
		// Surface tensions if option contact_angle=1
		RVect<dim> ForceTS1     = Model.force_TS1<dim>(lbmState);
		RVect<dim> ForceTS2     = Model.force_TS2<dim>(lbmState);
		RVect<dim> ForceTS3     = Model.force_TS3<dim>(lbmState);
		RVect<dim> ForceTS ;
		ForceTS[IX] = Model.contact_angle*(ForceTS1[IX]+ForceTS2[IX]+ForceTS3[IX]) + (1-Model.contact_angle)*ForceTS_2liq[IX] ;
		ForceTS[IY] = Model.contact_angle*(ForceTS1[IY]+ForceTS2[IY]+ForceTS3[IY]) + (1-Model.contact_angle)*ForceTS_2liq[IY] ;
		// Penalization force for moving structure interaction
		RVect<dim> ForcePenal  = Model.force_solid_impulsion<dim>(lbmState);
		// Gravity force
		RVect<dim> ForceG      = Model.force_G<dim>(lbmState);
		// Gravity force
		RVect<dim> ForceInjct  = Model.force_imposed_injct<dim>(lbmState);
		// Correction of pressure forces
		RVect<dim> ForceP      = Model.force_P_meth2<dim>(lbmState);
		RVect<dim> ForcePsolid = Model.force_P_solid<dim>(lbmState);
		// Viscosity force
		RVect<dim> ForceV;
		RVect<dim> ForceTot;

		// Calcul de la force correctrice de Fakhari et al
		//const real_t nu = Model.nu0 * Model.nu1 /
		//						(((1.0 - lbmState[IPHI]) * Model.nu1) + ((lbmState[IPHI]) * Model.nu0));
		//const real_t nu = Model.nu_NS_harm12(lbmState);
		
		const real_t nu = Model.nu_NS_harm12S(lbmState);
		real_t coeffV = - nu/(cs2 * dt);
		//const real_t eta = Model.rho0*Model.nu0 * Model.rho1*Model.nu1 /
		//	(((1.0 - lbmState[IPHI]) * Model.rho1*Model.nu1) + ((lbmState[IPHI])*Model.rho0*Model.nu0));
		//real_t coeffV = - eta/(lbmState[ID]*(cs2 * dt));
	
		collider.Calc_Lf();
		FState MinvSMg_geq = collider.Prod_invMSMf;

		// FORCE VISQUEUSE
		ForceV[IX] = 0.0;
		ForceV[IY] = 0.0;
		for (int beta = 0; beta < npop; ++beta) {
			ForceV[IX] +=
				E[beta][IX] * E[beta][IX] * (Model.rho1*lbmState[IDPHIDX]+Model.rho0*lbmState[IDPHI2DX]+Model.rho_sol*lbmState[IDPSIDX]) * MinvSMg_geq[beta] +
				E[beta][IX] * E[beta][IY] * (Model.rho1*lbmState[IDPHIDY]+Model.rho0*lbmState[IDPHI2DY]+Model.rho_sol*lbmState[IDPSIDY]) * MinvSMg_geq[beta] ;
			ForceV[IY] +=
				E[beta][IY] * E[beta][IX] * (Model.rho1*lbmState[IDPHIDX]+Model.rho0*lbmState[IDPHI2DX]+Model.rho_sol*lbmState[IDPSIDX]) * MinvSMg_geq[beta] +
				E[beta][IY] * E[beta][IY] * (Model.rho1*lbmState[IDPHIDY]+Model.rho0*lbmState[IDPHI2DY]+Model.rho_sol*lbmState[IDPSIDY]) * MinvSMg_geq[beta] ;
		}

		ForceV[IX] = coeffV * ForceV[IX];
		ForceV[IY] = coeffV * ForceV[IY];
		
		int dbb    = Model.diffuse_bounce_back ;
		real_t psi = lbmState[IPSI] ;
		ForcePsolid[IX] = (1-dbb)*ForcePsolid[IX] ;
		ForcePsolid[IY] = (1-dbb)*ForcePsolid[IY] ;
		// Calcul et enregistrement du terme source total
		ForceTot[IX] = (1.0 - psi)*(ForceG[IX]+ForceP[IX]+ForceTS[IX]+ForceV[IX]) + ForcePsolid[IX] + ForcePenal[IX];
		ForceTot[IY] = (1.0 - psi)*(ForceG[IY]+ForceP[IY]+ForceTS[IY]+ForceV[IY]) + ForcePsolid[IY] + ForcePenal[IY];

		this->set_lbm_val(IJK, IFX, ForceTot[IX]);
		this->set_lbm_val(IJK, IFY, ForceTot[IY]);
		real_t Sterm_mass_bal = Model.Sterm_solid_velocity_GradNum(lbmState);
		//real_t Sterm_mass_bal = Model.Sterm_solid_velocity_CT     (lbmState);
		// Rajout de la source dans l'équilibre
		
		real_t factr = dbb + (1-dbb)*(1.0 - psi) ;
		for (int ipop = 0; ipop < npop; ++ipop) {
			collider.S0[ipop] = dt * w[ipop] / (lbmState[ID] * cs2) *
			c * Base::compute_scal(ipop, ForceTot[IX], ForceTot[IY]) + w[ipop]*dt*Sterm_mass_bal ;
			collider.feq[ipop] = factr * collider.feq[ipop] - 0.5 * (collider.S0[ipop]);
		}
	}
	
	if (dim == 3) {
		// COMPUTATION OF Feq
		real_t scalUU = SQR(lbmState[IU]) + SQR(lbmState[IV]) + SQR(lbmState[IW]);
		for (int ipop = 0; ipop < npop; ++ipop) {
			real_t scalUC = c * Base::compute_scal(ipop, lbmState[IU], lbmState[IV], lbmState[IW]);
			GAMMA[ipop] = scalUC*cs2Inv + 0.5 * SQR(scalUC) * SQR(cs2Inv) - 0.5 * scalUU * cs2Inv;
			real_t feqbar =
				w[ipop] * (lbmState[IP] / (cs2 * lbmState[ID]) + GAMMA[ipop]);
			collider.feq[ipop] = feqbar;
			collider.f[ipop] = Base::get_f_val(tag, IJK, ipop);
		}

		// COMPUTATION OF MACROSCOPIC FORCE TERMS
		RVect<dim> ForceTS_2liq = Model.force_TS<dim>(lbmState);                 // Surface tension between 2 fluids
		RVect<dim> ForceTS ;
		if (Model.contact_angle == 1) {                                          // if option contact_angle=1
			RVect<dim> ForceTS1 = Model.force_TS1<dim>(lbmState);                // 	Surface tensions between phase 1 and phase 2
			RVect<dim> ForceTS2 = Model.force_TS2<dim>(lbmState);                // 	Surface tensions between phase 1 and solid phase
			RVect<dim> ForceTS3 = Model.force_TS3<dim>(lbmState);                // 	Surface tensions between phase 2 and solid phase
			ForceTS[IX] = ForceTS1[IX]+ForceTS2[IX]+ForceTS3[IX];
			ForceTS[IY] = ForceTS1[IY]+ForceTS2[IY]+ForceTS3[IY];
			ForceTS[IZ] = ForceTS1[IZ]+ForceTS2[IZ]+ForceTS3[IZ];
		} else {
			ForceTS[IX] = ForceTS_2liq[IX] ;
			ForceTS[IY] = ForceTS_2liq[IY] ;
			ForceTS[IZ] = ForceTS_2liq[IZ] ;
		}
		
		RVect<dim> ForceG      = Model.force_G<dim>(lbmState);                   // Gravity force
		RVect<dim> ForceP      = Model.force_P_meth2<dim>(lbmState);             // Correction of pressure forces
		RVect<dim> ForcePsolid = Model.force_P_solid<dim>(lbmState);
		RVect<dim> ForcePenal  = Model.force_solid_impulsion<dim>(lbmState);     // Penalization force for moving structure interaction
		//RVect<dim> ForceInjct  = Model.force_imposed_injct<dim>(lbmState);     // Test force injection
		RVect<dim> ForceTot;                                                     // Declaration total force
		RVect<dim> ForceV;                                                       // Declaration Viscosity force
		ForceV[IX] = 0.0;
		ForceV[IY] = 0.0;
		ForceV[IZ] = 0.0;
		const real_t nu = Model.nu_NS_harm12S(lbmState);
		real_t coeffV = - nu/(cs2 * dt);
		collider.Calc_Lf();
		FState MinvSMg_geq = collider.Prod_invMSMf;
		
		for (int beta = 0; beta < npop; ++beta) {
			ForceV[IX] +=
				E[beta][IX] * E[beta][IX] * (Model.rho1*lbmState[IDPHIDX]+Model.rho0*lbmState[IDPHI2DX]+Model.rho_sol*lbmState[IDPSIDX]) * MinvSMg_geq[beta] +
				E[beta][IX] * E[beta][IY] * (Model.rho1*lbmState[IDPHIDY]+Model.rho0*lbmState[IDPHI2DY]+Model.rho_sol*lbmState[IDPSIDY]) * MinvSMg_geq[beta] +
				E[beta][IX] * E[beta][IZ] * (Model.rho1*lbmState[IDPHIDZ]+Model.rho0*lbmState[IDPHI2DZ]+Model.rho_sol*lbmState[IDPSIDZ]) * MinvSMg_geq[beta];
			ForceV[IY] +=
				E[beta][IY] * E[beta][IX] * (Model.rho1*lbmState[IDPHIDX]+Model.rho0*lbmState[IDPHI2DX]+Model.rho_sol*lbmState[IDPSIDX]) * MinvSMg_geq[beta] +
				E[beta][IY] * E[beta][IY] * (Model.rho1*lbmState[IDPHIDY]+Model.rho0*lbmState[IDPHI2DY]+Model.rho_sol*lbmState[IDPSIDY]) * MinvSMg_geq[beta] +
				E[beta][IY] * E[beta][IZ] * (Model.rho1*lbmState[IDPHIDZ]+Model.rho0*lbmState[IDPHI2DZ]+Model.rho_sol*lbmState[IDPSIDZ]) * MinvSMg_geq[beta];
			ForceV[IZ] +=
				E[beta][IZ] * E[beta][IX] * (Model.rho1*lbmState[IDPHIDX]+Model.rho0*lbmState[IDPHI2DX]+Model.rho_sol*lbmState[IDPSIDX]) * MinvSMg_geq[beta] +
				E[beta][IZ] * E[beta][IY] * (Model.rho1*lbmState[IDPHIDY]+Model.rho0*lbmState[IDPHI2DY]+Model.rho_sol*lbmState[IDPSIDY]) * MinvSMg_geq[beta] +
				E[beta][IZ] * E[beta][IZ] * (Model.rho1*lbmState[IDPHIDZ]+Model.rho0*lbmState[IDPHI2DZ]+Model.rho_sol*lbmState[IDPSIDZ]) * MinvSMg_geq[beta];
		}

		ForceV[IX] = coeffV * ForceV[IX];
		ForceV[IY] = coeffV * ForceV[IY];
		ForceV[IZ] = coeffV * ForceV[IZ];

		int dbb    = Model.diffuse_bounce_back ;
		real_t psi = lbmState[IPSI] ;
		ForcePsolid[IX] = (1-dbb)*ForcePsolid[IX] ;
		ForcePsolid[IY] = (1-dbb)*ForcePsolid[IY] ;
		ForcePsolid[IZ] = (1-dbb)*ForcePsolid[IZ] ;
		// Calcul et enregistrement du terme source total
		ForceTot[IX] = (1.0 - psi)*(ForceG[IX]+ForceP[IX]+ForceTS[IX]+ForceV[IX]) + ForcePsolid[IX] + ForcePenal[IX];
		ForceTot[IY] = (1.0 - psi)*(ForceG[IY]+ForceP[IY]+ForceTS[IY]+ForceV[IY]) + ForcePsolid[IY] + ForcePenal[IY];
		ForceTot[IZ] = (1.0 - psi)*(ForceG[IZ]+ForceP[IZ]+ForceTS[IZ]+ForceV[IZ]) + ForcePsolid[IZ] + ForcePenal[IZ];
		
		this->set_lbm_val(IJK, IFX, ForceTot[IX]);
		this->set_lbm_val(IJK, IFY, ForceTot[IY]);
		this->set_lbm_val(IJK, IFZ, ForceTot[IZ]);
		// Rajout de la source dans l'équilibre
		
		real_t Sterm_mass_bal = Model.Sterm_solid_velocity_GradNum(lbmState);
		//real_t Sterm_mass_bal = Model.Sterm_solid_velocity_CT     (lbmState);		
		real_t factr = dbb + (1-dbb)*(1.0 - psi) ;
		for (int ipop = 0; ipop < npop; ++ipop) {
			collider.S0[ipop] =
				dt * w[ipop] / (lbmState[ID] * cs2) *
			c * Base::compute_scal(ipop, ForceTot[IX], ForceTot[IY], ForceTot[IZ]);
			collider.feq[ipop] = factr * collider.feq[ipop] - 0.5 * (collider.S0[ipop]);
		}
	}
  }
  //
  //
  // ================================== NS TRT COLLISION =====================================
  // SETUP_COLLIDER FOR NS - Fakhari et. al. model (2017)
  //
  //
  KOKKOS_INLINE_FUNCTION
  void setup_collider(EquationTag2 tag, const IVect<dim> &IJK,
                      TRTCollider<dim, npop> &collider) const {

    // Paramètres pour la simulation
    const real_t dx = Model.dx;
    const real_t dt = Model.dt;
    const real_t c = dx / dt;
    const real_t cs2 = SQR(c) / Model.e2;

    // Stockage des anciennes grandeurs macroscopiques
    LBMState lbmState;
    Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmState);

    // compute collision rate
    collider.tauA = Model.tau(lbmState);
    collider.tauS = 0.5 + this->params.lambdaTRT2 / (collider.tauA - 0.5);
    // real_t staudx = 1/((collider.tauA-0.5)*dx/Model.e2);

    // Calcul de l'équilibre sans terme source
    FState GAMMA;

    if (dim == 2) {
      real_t scalUU = SQR(lbmState[IU]) + SQR(lbmState[IV]);
      for (int ipop = 0; ipop < npop; ++ipop) {
        real_t scalUC =
            c * Base::compute_scal(ipop, lbmState[IU], lbmState[IV]);
        GAMMA[ipop] =
            scalUC / cs2 + 0.5 * SQR(scalUC) / SQR(cs2) - 0.5 * scalUU / cs2;
        real_t feqbar =
            w[ipop] * (lbmState[IP] / (cs2 * lbmState[ID]) + GAMMA[ipop]);
        collider.feq[ipop] = feqbar;
        collider.f[ipop] = Base::get_f_val(tag, IJK, ipop);
      }

      // Calculs des termes forces pour le terme source

      // RVect<dim> gradrho = Model.grad_rho<dim>(lbmState);
      RVect<dim> ForceTS = Model.force_TS<dim>(lbmState);

      RVect<dim> ForceG = Model.force_G<dim>(lbmState);
      RVect<dim> ForceP = Model.force_P_meth1<dim>(lbmState);
      RVect<dim> ForceV;
      RVect<dim> ForceTot;

      // Calcul de la force correctrice de Fakhari et al
      const real_t nu = Model.nu0 * Model.nu1 /
                        (((1.0 - lbmState[IPHI]) * Model.nu1) +
                         ((lbmState[IPHI]) * Model.nu0));
      real_t coeffV =
          -(nu * (Model.rho1 - Model.rho0)) / (cs2 * dt * collider.tauA);

      ForceV[IX] = 0.0;
      ForceV[IY] = 0.0;

      for (int alpha = 0; alpha < npop; ++alpha) {
        // real_t f_TRT = collider.get_fTRT(alpha);

        ForceV[IX] += E[alpha][IX] * E[alpha][IX] * lbmState[IDRHODX] *
                          collider.get_fTRT(alpha) +
                      E[alpha][IX] * E[alpha][IY] * lbmState[IDRHODY] *
                          collider.get_fTRT(alpha);
        ForceV[IY] += E[alpha][IY] * E[alpha][IX] * lbmState[IDRHODX] *
                          collider.get_fTRT(alpha) +
                      E[alpha][IY] * E[alpha][IY] * lbmState[IDRHODY] *
                          collider.get_fTRT(alpha);
      }

      ForceV[IX] = coeffV * ForceV[IX];
      ForceV[IY] = coeffV * ForceV[IY];

      // Calcul et enregistrement du terme source total
      ForceTot[IX] = ForceG[IX] + ForceP[IX] + ForceTS[IX] + ForceV[IX];
      ForceTot[IY] = ForceG[IY] + ForceP[IY] + ForceTS[IY] + ForceV[IY];

      this->set_lbm_val(IJK, IFX, ForceTot[IX]);
      this->set_lbm_val(IJK, IFY, ForceTot[IY]);

      // Rajout de la source dans l'équilibre
      for (int ipop = 0; ipop < npop; ++ipop) {
        collider.S0[ipop] =
            dx * w[ipop] / (lbmState[ID] * cs2) *
            Base::compute_scal(ipop, ForceTot[IX], ForceTot[IY]);
        collider.feq[ipop] = collider.feq[ipop] - 0.5 * (collider.S0[ipop]);
      }
    }

    if (dim == 3) {
      real_t scalUU = SQR(lbmState[IU]) + SQR(lbmState[IV]) + SQR(lbmState[IW]);
      for (int ipop = 0; ipop < npop; ++ipop) {
        real_t scalUC = c * Base::compute_scal(ipop, lbmState[IU], lbmState[IV],
                                               lbmState[IW]);
        GAMMA[ipop] =
            scalUC / cs2 + 0.5 * SQR(scalUC) / SQR(cs2) - 0.5 * scalUU / cs2;
        real_t feqbar =
            w[ipop] * (lbmState[IP] / (cs2 * lbmState[ID]) + GAMMA[ipop]);
        collider.feq[ipop] = feqbar;
        collider.f[ipop] = Base::get_f_val(tag, IJK, ipop);
      }

      // Calculs des termes forces pour le terme source

      // RVect<dim> gradrho = Model.grad_rho<dim>(lbmState);
      RVect<dim> ForceTS = Model.force_TS<dim>(lbmState);

      RVect<dim> ForceG = Model.force_G<dim>(lbmState);
      RVect<dim> ForceP = Model.force_P_meth1<dim>(lbmState);
      RVect<dim> ForceV;
      RVect<dim> ForceTot;

      // Calcul de la force correctrice de Fakhari et al
      const real_t nu = Model.nu0 * Model.nu1 /
                        (((1.0 - lbmState[IPHI]) * Model.nu1) +
                         ((lbmState[IPHI]) * Model.nu0));
      real_t coeffV =
          -(nu * (Model.rho1 - Model.rho0)) / (cs2 * dt * collider.tauA);

      ForceV[IX] = 0.0;
      ForceV[IY] = 0.0;
      ForceV[IZ] = 0.0;

      for (int alpha = 0; alpha < npop; ++alpha) {
        // real_t f_TRT = collider.get_fTRT(alpha);

        ForceV[IX] += E[alpha][IX] * E[alpha][IX] * lbmState[IDRHODX] *
                          collider.get_fTRT(alpha) +
                      E[alpha][IX] * E[alpha][IY] * lbmState[IDRHODY] *
                          collider.get_fTRT(alpha) +
                      E[alpha][IX] * E[alpha][IZ] * lbmState[IDRHODZ] *
                          collider.get_fTRT(alpha);
        ForceV[IY] += E[alpha][IY] * E[alpha][IX] * lbmState[IDRHODX] *
                          collider.get_fTRT(alpha) +
                      E[alpha][IY] * E[alpha][IY] * lbmState[IDRHODY] *
                          collider.get_fTRT(alpha) +
                      E[alpha][IY] * E[alpha][IZ] * lbmState[IDRHODZ] *
                          collider.get_fTRT(alpha);
        ForceV[IZ] += E[alpha][IZ] * E[alpha][IX] * lbmState[IDRHODX] *
                          collider.get_fTRT(alpha) +
                      E[alpha][IZ] * E[alpha][IY] * lbmState[IDRHODY] *
                          collider.get_fTRT(alpha) +
                      E[alpha][IZ] * E[alpha][IZ] * lbmState[IDRHODZ] *
                          collider.get_fTRT(alpha);
      }

      ForceV[IX] = coeffV * ForceV[IX];
      ForceV[IY] = coeffV * ForceV[IY];
      ForceV[IZ] = coeffV * ForceV[IZ];

      // Calcul et enregistrement du terme source total
      ForceTot[IX] = ForceG[IX] + ForceP[IX] + ForceTS[IX] + ForceV[IX];
      ForceTot[IY] = ForceG[IY] + ForceP[IY] + ForceTS[IY] + ForceV[IY];
      ForceTot[IZ] = ForceG[IZ] + ForceP[IZ] + ForceTS[IZ] + ForceV[IZ];

      this->set_lbm_val(IJK, IFX, ForceTot[IX]);
      this->set_lbm_val(IJK, IFY, ForceTot[IY]);
      this->set_lbm_val(IJK, IFZ, ForceTot[IZ]);

      // Rajout de la source dans l'équilibre
      for (int ipop = 0; ipop < npop; ++ipop) {
        collider.S0[ipop] =
            dt * w[ipop] / (lbmState[ID] * cs2) *
            Base::compute_scal(ipop, ForceTot[IX], ForceTot[IY], ForceTot[IZ]);
        collider.feq[ipop] = collider.feq[ipop] - 0.5 * (collider.S0[ipop]);
      }
    }
  } // end of setup_collider for composition equation

  // ==================== COMPOSITION EQUATION ===========================

  KOKKOS_INLINE_FUNCTION
  void setup_collider(EquationTag3 tag, const IVect<dim> &IJK,
                      BGK_Collider &collider) const {
    // Paramètres pour la simulation
    const real_t dt = Model.dt;
    const real_t dx = Model.dx;
    const real_t e2 = Model.e2;
    const real_t c = dx / dt;
    // const real_t cs2 = SQR(c)/e2;
    const real_t c_cs2 = e2 / c;

    // Stockage des anciennes grandeurs macroscopiques
    LBMState lbmState;
    Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmState);

    const real_t M0 = Model.M0_C(lbmState);
    const RVect<dim> M1 = Model.M1_C<dim>(lbmState);
    const real_t M2 = Model.M2C_C(lbmState);

    // compute collision rate
    collider.tau = Model.tau_C(type, lbmState);

    // ipop = 0
    collider.f[0] = Base::get_f_val(tag, IJK, 0);
    //collider.S0[0] = w[0] * dx * Base::compute_scal(0, S0);
    collider.S0[0] = w[0] * dt * Model.St_C (lbmState) ;
    collider.feq[0] = M0 - (1 - w[0]) * M2 - 0.5 * collider.S0[0];

    // ipop > 0
    for (int ipop = 1; ipop < npop; ++ipop) {
      collider.f[ipop] = this->get_f_val(tag, IJK, ipop);
      //collider.S0[ipop] = w[ipop] * dx * Base::compute_scal(ipop, S0);
      collider.S0[ipop] = w[ipop] * dt * Model.St_C (lbmState) ;
      collider.feq[ipop] =
          w[ipop] * (M2 + c_cs2 * Base::compute_scal(ipop, M1)) -
          0.5 * collider.S0[ipop];
    }
  }

  KOKKOS_INLINE_FUNCTION
  void setup_collider(EquationTag3 tag, const IVect<dim> &IJK,
                      MRT_Collider &collider) const {
    // Setup matrices MRT
    // collider.setMandMinv(Base::M, Base::Minv);

    // Paramètres pour la simulation
    const real_t dt = Model.dt;
    const real_t dx = Model.dx;
    const real_t e2 = Model.e2;
    const real_t c = dx / dt;
    // const real_t cs2 = SQR(c)/e2;
    const real_t c_cs2 = e2 / c;

    // Stockage des anciennes grandeurs macroscopiques
    LBMState lbmState;
    Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmState);

    const real_t M0 = Model.M0_C(lbmState);
    const RVect<dim> M1 = Model.M1_C<dim>(lbmState);
    const real_t M2 = Model.M2C_C(lbmState);
    const RVect<dim> S0 = Model.S0_C<dim>(type, lbmState);

    // compute collision rate
    collider.tau = Model.tau_C(type, lbmState);
    const real_t tauInv = 1 / collider.tau;

    // Remplissage de la matrice S
    for (int i = 0; i < npop; i++) {
      if (Model.tauMatrixComp[i] != -1.0) {
        collider.S[i] = Model.tauMatrixComp[i];
      } else {
        collider.S[i] = tauInv;
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
      collider.feq[ipop] =
          w[ipop] * (M2 + c_cs2 * Base::compute_scal(ipop, M1)) -
          0.5 * collider.S0[ipop];
    }
  }

  KOKKOS_INLINE_FUNCTION
  void setup_collider(EquationTag3 tag, const IVect<dim> &IJK,
                      TRTCollider<dim, npop> &collider) const {


    const real_t dt = this->params.dt;
    const real_t dx = this->params.dx;

    LBMState lbmState;
    this->setupLBMState(IJK, lbmState);

    const real_t M0 = Model.M0(lbmState);
    const real_t M2 = Model.M2(lbmState);
    const real_t S0 = Model.S_stt(lbmState) + Model.S_dwt(lbmState);
    const RVect<dim> S1 = Model.S_ct<dim>(lbmState);

    // compute collision rate
    collider.tauA = Model.tau(lbmState);
    collider.tauS = 0.5 + this->params.lambdaTRT2 / (collider.tauA - 0.5);
    real_t staudx = 1 / ((collider.tauA - 0.5) * dx / Model.e2);

    // ipop = 0
    collider.f[0] = this->get_f_val(tag, IJK, 0);
    real_t scal = this->compute_scal(0, S1);
    collider.S0[0] = dt * w[0] * (S0 + staudx * scal);
    //~ collider.S1[0] = dt * w[0] * staudx * scal;
    collider.feq[0] = M0 - (1 - w[0]) * M2 - 0.5 * collider.S0[0];

    // ipop > 0
    for (int ipop = 1; ipop < npop; ++ipop) {
      collider.f[ipop] = this->get_f_val(tag, IJK, ipop);
      scal = this->compute_scal(ipop, S1);
      collider.S0[ipop] = dt * w[ipop] * (S0 + staudx * scal);
      //~ collider.S1[ipop] = dt * w[ipop] * staudx * scal;
      collider.feq[ipop] = w[ipop] * M2 - 0.5 * (collider.S0[ipop]);
    }
  } // end of setup_collider for composition equation

  // ==================== PHASE FIELD EQUATION PSI =======================

  // ====================       ONLY  BGK          =======================
  KOKKOS_INLINE_FUNCTION
  void setup_collider(EquationTag4 tag, const IVect<dim> &IJK,
                      BGK_Collider_Time_Factor& collider) const {
	
	real_t x,y;
	this->get_coordinates(IJK, x, y);

	const real_t dt   = Model.dt;
	const real_t dx   = Model.dx;
	const real_t e2   = Model.e2;
	const real_t cs2  = 1.0/e2 * SQR(dx/dt);
	const real_t W0   = Model.KR_W0;
	const real_t tau0 = Model.KR_tau0;
	const real_t fact = W0*W0/tau0;
	const int CT      = Model.KR_counter_term ;

	LBMState lbmState;
	Base::template setupLBMState2<LBMState,COMPONENT_SIZE>(IJK, lbmState);
	bool can_str ;
	// Moment 0: PSI
	const real_t M0 = Model.M0_PSI(tag, lbmState);
	// Counter term
	const RVect<dim> G_ct = Model.S_ct_psi<dim>(lbmState);
	// Anisotropy coefficient a_s(n) and Anisotropy vector N
	const real_t As_n = Model.compute_anisotropy<dim>(tag, lbmState);
	real_t As2 = As_n*As_n;
	const RVect<dim> N = Model.compute_anisotropy_vector<dim>(tag, lbmState);
	real_t staudx = CT * 1/((Model.tau_PSI(tag, lbmState,As2)-0.5)*dx/e2);
	// Source term for psi eq
	const real_t PSI_St = (1-CT)*Model.Sdw_psi (tag,lbmState) + Model.St_psi (tag, lbmState);
	IVect<dim> IJKs;
	can_str = this->stream_alldir(IJK,IJKs,0);
	if (can_str) 	{collider.f_nonlocal[0] = this->get_f_val(tag,IJKs,0);}
	else 			{collider.f_nonlocal[0] = 0.0;}

	// compute collision rate
	collider.tau = Model.tau_PSI(tag, lbmState, As2);
	
	// ipop = 0
	collider.f[0]   = Base::get_f_val(tag,IJK,0);
	real_t scal     = Base::compute_scal(0, G_ct);
	collider.S0[0]  = w[0]*dt*(PSI_St + staudx * scal);
	collider.feq[0] = w[0]*(M0-e2*Base::compute_scal(0,N)*(dt/dx)*fact) - 0.5 * collider.S0[0];
	collider.factor = As2;
	// ipop > 0
	for (int ipop=1; ipop<npop; ++ipop) {
		collider.f[ipop]   = this->get_f_val(tag,IJK,ipop);
		collider.S0[ipop]  = dt*w[ipop]*(PSI_St + staudx * Base::compute_scal(ipop, G_ct));
		collider.feq[ipop] = w[ipop]*(M0-e2*Base::compute_scal(ipop,N)*(dt/dx)*W0*W0/tau0) - 0.5 * collider.S0[ipop];
		can_str = this->stream_alldir(IJK,IJKs,ipop);
		if (can_str) 	{collider.f_nonlocal[ipop] = this->get_f_val(tag,IJKs,ipop);}
		else 			{collider.f_nonlocal[ipop] = 0.0;}
	}
  } // end of setup_collider for phase field equation PSI for crystal growth
  // =====================================================================
  //                        MAKE BOUNDARY
  // =====================================================================

  KOKKOS_INLINE_FUNCTION
  void make_boundary(const IVect<dim> &IJK, int faceId) const {

    const real_t dx = this->params.dx;
    const real_t dt = this->params.dt;
    const real_t c = dx / dt;
    // const real_t cs2 = SQR(c)/Model.e2;
    const real_t c_cs2 = Model.e2 / (c * Model.rho0); // Ratio c/(cs2*rho0)

    // phi boundary
    if (Base::params.boundary_types[BOUNDARY_EQUATION_1][faceId] ==
        BC_ANTI_BOUNCE_BACK) {
      real_t boundary_value =
          Base::params.boundary_values[BOUNDARY_PHASE_FIELD][faceId];
      for (int ipop = 0; ipop < npop; ++ipop)
        this->compute_boundary_antibounceback(tagPHI, faceId, IJK, ipop,
                                              boundary_value);

    }

    else if (Base::params.boundary_types[BOUNDARY_EQUATION_1][faceId] ==
             BC_ZERO_FLUX) {
      for (int ipop = 0; ipop < npop; ++ipop)
        this->compute_boundary_bounceback(tagPHI, faceId, IJK, ipop, 0.0);
    }
    
    // psi boundary
    if (Base::params.boundary_types[BOUNDARY_EQUATION_2][faceId] ==
        BC_ANTI_BOUNCE_BACK) {
      real_t boundary_value =
          Base::params.boundary_values[BOUNDARY_PHASE_FIELD][faceId];
      for (int ipop = 0; ipop < npop; ++ipop)
        this->compute_boundary_antibounceback(tagPSI, faceId, IJK, ipop,
                                              boundary_value);

    }

    else if (Base::params.boundary_types[BOUNDARY_EQUATION_2][faceId] ==
             BC_ZERO_FLUX) {
      for (int ipop = 0; ipop < npop; ++ipop)
        this->compute_boundary_bounceback(tagPSI, faceId, IJK, ipop, 0.0);
    }

    // NS boundaries
    if (Base::params.boundary_types[BOUNDARY_EQUATION_3][faceId] ==
        BC_ANTI_BOUNCE_BACK) {
      real_t boundary_value =
          this->params.boundary_values[BOUNDARY_PRESSURE][faceId];
      for (int ipop = 0; ipop < npop; ++ipop) {
        this->compute_boundary_antibounceback(tagNS, faceId, IJK, ipop,
                                              boundary_value);
      }
    }

    else if (Base::params.boundary_types[BOUNDARY_EQUATION_3][faceId] ==
             BC_ZERO_FLUX) {
      for (int ipop = 0; ipop < npop; ++ipop)
        this->compute_boundary_bounceback(tagNS, faceId, IJK, ipop, 0.0);

    }

    else if (Base::params.boundary_types[BOUNDARY_EQUATION_3][faceId] ==
             BC_BOUNCEBACK) {

      real_t boundary_vx =
          this->params.boundary_values[BOUNDARY_VELOCITY_X][faceId];
      real_t boundary_vy =
          this->params.boundary_values[BOUNDARY_VELOCITY_Y][faceId];

      real_t value;
      for (int ipop = 0; ipop < npop; ++ipop) {
        value = c_cs2 * this->compute_scal_Ebar(ipop, boundary_vx, boundary_vy);
        this->compute_boundary_bounceback(tagNS, faceId, IJK, ipop, value);
      }

    } else if (Base::params.boundary_types[BOUNDARY_EQUATION_3][faceId] ==
               BC_POISEUILLE) {
      const real_t isize = this->params.isize;
      const real_t jsize = this->params.jsize;

      //~ const real_t cs = SQRT(cs2);
      real_t scaling = (faceId == FACE_YMIN or faceId == FACE_YMAX) * 4 *
                           IJK[IX] * (isize - IJK[IX]) / SQR(isize) +
                       (faceId == FACE_XMIN or faceId == FACE_XMAX) * 4 *
                           IJK[IY] * (jsize - IJK[IY]) / SQR(jsize);
      real_t boundary_vx =
          scaling * this->params.boundary_values[BOUNDARY_VELOCITY_X][faceId];
      real_t boundary_vy =
          scaling * this->params.boundary_values[BOUNDARY_VELOCITY_Y][faceId];

      real_t value;
      for (int ipop = 0; ipop < npop; ++ipop) {
        value = c_cs2 * this->compute_scal_Ebar(ipop, boundary_vx, boundary_vy);
        this->compute_boundary_bounceback(tagNS, faceId, IJK, ipop, value);
      }
    }

    // Composition boundaries
    if (Base::params.boundary_types[BOUNDARY_EQUATION_4][faceId] ==
        BC_ANTI_BOUNCE_BACK) {
      real_t boundary_value =
          Base::params.boundary_values[BOUNDARY_CONCENTRATION][faceId];
      for (int ipop = 0; ipop < npop; ++ipop)
        this->compute_boundary_antibounceback(tagCOMP, faceId, IJK, ipop,
                                              boundary_value);

    }

    else if (Base::params.boundary_types[BOUNDARY_EQUATION_4][faceId] ==
             BC_ZERO_FLUX) {
      for (int ipop = 0; ipop < npop; ++ipop)
        this->compute_boundary_bounceback(tagCOMP, faceId, IJK, ipop, 0.0);
    }
  };

  // =====================================================================
  //                            INIT MACRO
  // =====================================================================

  KOKKOS_INLINE_FUNCTION
  void init_macro(IVect<dim> IJK, RANDOM_POOL::generator_type rand_gen) const {

    real_t x, y, z;
    if (dim == 2) {
      // get local coordinates
      this->get_coordinates(IJK, x, y);
    }

    else if (dim == 3) {
      this->get_coord3D(IJK, x, y, z);
    }
	const real_t pi = M_PI;
    // get useful params
    real_t xphi    = 0.0;
    real_t xc      = 0.0;
	real_t xpsi    = 0.0;
	real_t psi     = 0.0;
    real_t phi     = 0.0;
    real_t Sinject = 0.0;
    real_t rho     = 0.0;
    real_t vx      = 0.0;
    real_t vy      = 0.0;
    real_t vz      = 0.0;
    real_t p       = 0.0;
    real_t c       = 0.0;
    real_t mu      = 0.0;

	// ****************************************************************************************************
	//
	//                                INITIALIZATION OF FLUID PHASE (PHI)
	//
	// ****************************************************************************************************

    if (Model.initType == PHASE_FIELD_INIT_VERTICAL) {
		xphi = x - Model.x0;
		xc   = xphi ;
	}
    else if (Model.initType == PHASE_FIELD_INIT_SPHERE) {
		if (dim == 2) {
			xphi = (Model.r0 - sqrt(SQR(x - Model.x0) + SQR(y - Model.y0)));
		}
		else if (dim == 3) {
			xphi = (Model.r0 - sqrt(SQR(x - Model.x0) + SQR(y - Model.y0) + SQR(z - Model.z0)));
		}
		xc = xphi ;
	}
	else if (Model.initType == PHASE_FIELD_INIT_CYLINDER) {
		if (dim == 2) {
			xphi = (Model.r0 - sqrt(SQR(x - Model.x0) + SQR(y - Model.y0)));
		}
		else if (dim == 3) {
			real_t d1 = Model.r0 - sqrt(SQR(x - Model.x0) + SQR(y - Model.y0)) ;
			real_t d2 = Model.r1 - sqrt(SQR(z - Model.z0)) ;
			xphi = FMIN(d1,d2);
		}
		xc = xphi ;
	}
	else if (Model.initType == PHASE_FIELD_INIT_RECTANGLE){
		if (dim == 2) xphi = Model.phi_init_Rectangle2D(x, y) ;
	}
    else if (Model.initType == PHASE_FIELD_INIT_2SPHERE) {
      real_t d = sqrt(SQR(x - Model.x0) + SQR(y - Model.y0));
      bool ri = (d > ((Model.r0 + Model.r1) / 2));
      xphi = ri * (Model.r0 - d) - (1 - ri) * (Model.r1 - d);
      xc   = xphi ;
    }

    else if (Model.initType == PHASE_FIELD_INIT_SQUARE) {
		xphi = FMIN(Model.r0 - sqrt(SQR(x - Model.x0)),
                  Model.r1 - sqrt(SQR(y - Model.y0)));
		xc   = xphi ;
	}
    else if (Model.initType == PHASE_FIELD_INIT_MIXTURE) {
		xphi = x - Model.x0;
		xc   = xphi ;
	}
    else if (Model.initType == PHASE_FIELD_INIT_TAYLOR) {

      if (dim == 2) {
        real_t L = Model.wave_length;
        xphi =
            y -
            Model.ampl * sin(Model.n_wave * 2.0 * 3.14 / L * x + 3.14 / 2.0) -
            Model.height;
        xc = y -
             Model.ampl * sin(Model.n_wave * 2.0 * 3.14 / L * x + 3.14 / 2.0) -
             Model.height;
      }

      else if (dim == 3) {
        real_t L = Model.wave_length;
        xphi = z -
               Model.ampl * (cos(Model.n_wave * x * 2.0 * 3.14 / L) +
                             cos(Model.n_wave * y * 2.0 * 3.14 / L)) -
               Model.height;

        xc = xphi;
      }
    }
	
	else if (Model.initType == TWO_PHASE_BREAKING_WAVE) {
		if (dim == 2) {
			real_t L  = Model.wave_length;
			real_t k  = 2.0*pi/L;
			real_t a  =  Model.ampl;
			real_t g  = -Model.gy ;
			real_t omega = sqrt(g*k*(1.0+SQR(a*k))) ;
			xphi      = y -
						(a*cos(k*x) + 0.5*(a*a)*k*cos(2.0*k*x) + 0.375*a*a*a*k*k*cos(3.0*k*x)) -
							Model.height;
			xc        = xphi ;
		}
		else if (dim == 3) {
			real_t L  = Model.wave_length;
			real_t k  = 2.0*pi/L;
			real_t a  = Model.ampl;
			xphi      = z -
						(a*cos(k*x) + 0.5*(a*a)*k*cos(2.0*k*x) + 0.375*a*a*a*k*k*cos(3.0*k*x)) -
							Model.height;
			xc        = xphi ;
      }
	}

    
    else if (Model.initType == TWO_PLANE) {
      real_t L = Model.wave_length;
      xphi = y -
             Model.ampl * sin(Model.n_wave * 2.0 * 3.14 / L * x + 3.14 / 2.0) -
             Model.height;
      xc = y -
           Model.ampl * sin(Model.n_wave * 2.0 * 3.14 / L * x + 3.14 / 2.0) -
           Model.height;
      vx = 0.5 * (Model.initVX_upper + Model.initVX_lower +
                  (Model.initVX_upper - Model.initVX_lower) *
                      tanh(Model.sign * 2.0 * xphi / Model.W));
      c = Model.c0(xc);
    }
    
    else if (Model.initType == BUBBLE_TAYLOR) {
		if (dim == 2) {
			real_t L = Model.wave_length;
			real_t d1, d2;
			d1 = y -
				Model.ampl * sin(Model.n_wave * 2.0 * 3.14 / L * x + 3.14 / 2.0) -
				Model.height;
			d2 = Model.R - sqrt(SQR(x - Model.xC) + SQR(y - Model.yC));
			xphi = abs(d1) > abs(d2) ? d2 : d1;
			// xphi = d1;
			xc = xphi;
		}

		if (dim == 3) {
			// real_t L = Model.wave_length;
			real_t d1, d2;
			d1 = Model.height - sqrt(SQR(z - Model.z0));
			d2 = Model.R - sqrt(SQR(x - Model.xC) + SQR(y - Model.yC));
			xphi = FMIN(d1, d2);
			xc = xphi;
		}
    }
	else if (Model.initType == TWO_PHASE_SERPENTINE) {
      if (dim == 2) xphi = (Model.R - sqrt(SQR(x - Model.xC) + SQR(y - Model.yC)));
    }
    
    else if (Model.initType == TWO_PHASE_ZALESAK) {
      if (dim == 2) xphi = (Model.R - sqrt(SQR(x - Model.xC) + SQR(y - Model.yC)));
    }
    
    else if (Model.initType == RISING_BUBBLE) {
	  if (dim == 2) {
        real_t d1, d2;
        d1 = Model.R -
             sqrt(SQR(x - Model.xC) + SQR(y - Model.yC)) ;
        d2 = Model.height - y;
        xphi = abs(d1) > abs(d2) ? d2 : d1;
      }
      if (dim == 3) {
        real_t d1, d2;
        d1 = Model.R -
             sqrt(SQR(x - Model.xC) + SQR(y - Model.yC) + SQR(z - Model.zC));
        d2 = Model.z0 - z;
        xphi = abs(d1) > abs(d2) ? d2 : d1;
      }
    }
    
    else if (Model.initType == TWO_PHASE_SPLASH) {
		if (dim == 2) {
			real_t d1, d2;
			d1 = Model.R - sqrt(SQR(x - Model.xC) + SQR(y - Model.yC)) ;
			d2 = Model.height - y;
			xphi = abs(d1) > abs(d2) ? d2 : d1;
		}
		else if (dim == 3) {
			real_t d1, d2;
			d1 = Model.R -
				sqrt(SQR(x - Model.xC) + SQR(y - Model.yC) + SQR(z - Model.zC));
			d2 = Model.z0 - z;
			xphi = abs(d1) > abs(d2) ? d2 : d1;
		}
    }
	
	else if (Model.initType == DAM_BREAK) {
	  if (dim == 2) {
		real_t d1;
		d1 = FMIN(Model.width_R - sqrt(SQR(x-Model.x0)), Model.height_R - sqrt(SQR(y-Model.y0)));
		//d2 = Model.height - y;
		xphi = d1;
		//xphi = abs(d1) > abs(d2) ? d2 : d1;
      }
      if (dim == 3) {
        xphi = FMIN(Model.width_R - sqrt(SQR(x-Model.x0)), Model.height_R - sqrt(SQR(y-Model.y0)));
      }
    }
    
    else if (Model.initType == PHASE_FIELD_INIT_TAYLOR_PERTURB_EXPO) {
		if (dim == 2) {
			xphi = y - (Model.height +
						Model.ampl * exp(-SQR(x) / (2 * SQR(Model.sigma0))));
			xc = xphi;
		} else if (dim == 3) {
			xphi = z - (Model.height +
						Model.ampl * exp(-SQR(x) / (2 * SQR(Model.sigma0))) +
						Model.ampl * exp(-SQR(y) / (2 * SQR(Model.sigma0))));
			xc = xphi;
		}
	}
//
//  CONTAINER FILLED WITH FLUID
//
	else if (Model.initType == PHASE_FIELD_GLASS) {
		real_t xphi_1 = (-Model.r0 + sqrt(SQR(x - Model.x0) + SQR(y - Model.y0)));
		real_t x0 = Model.x0_gl;
		real_t y0 = Model.y0_gl - Model.height_gl/2 + Model.thickness_gl + Model.h_glass/2;
		real_t l = Model.width_gl - 2* Model.thickness_gl;
		real_t L = Model.h_glass;
		real_t xphi_2;
		
		if ( sqrt(SQR(x-x0)) <= l/2 or sqrt(SQR(y-y0)) <= L/2 ){
			xphi_2 = FMAX(sqrt(SQR(x0-x))-l/2, sqrt(SQR(y0-y))-L/2);
		} else {
			if (y > y0 + L/2){
				if(x < x0 -l/2){
					xphi_2 = sqrt(SQR(x - (x0-l/2)) + SQR(y - (y0+L/2)));
				}
				if(x > x0 + l/2){
					xphi_2 = sqrt(SQR(x - (x0+l/2)) + SQR(y - (y0+L/2)));
				}
			}
			if (y < y0 - L/2){
				if(x < x0 -l/2){
					xphi_2 = sqrt(SQR(x - (x0-l/2)) + SQR(y - (y0-L/2)));
				}
				if(x > x0 + l/2){
					xphi_2 = sqrt(SQR(x - (x0+l/2)) + SQR(y - (y0-L/2)));
				}
			}
		}
	
		xphi = FMIN(xphi_1,xphi_2);		
	}
//
//  TUBE DE TORRICELLI
//
	else if (Model.initType == PHASE_FIELD_TORRICELLI){
        
		//real_t x0_tu = Model.x0_gl;
        real_t y0_tu = Model.y0_tu;
        real_t height_tu = Model.height_tu;
        real_t width_tu = Model.width_tu;
        real_t thickness_tu = Model.thickness_tu;
        real_t x0_gl = Model.x0_gl;
        real_t y0_gl = Model.y0_gl;
        real_t height_gl = Model.height_gl;
        real_t width_gl = Model.width_gl;
        real_t thickness_gl = Model.thickness_gl;
        real_t h_glass = Model.h_glass;
        real_t x0 = x0_gl;
        real_t y0 = (y0_gl + thickness_gl + y0_tu) /2 - (height_gl + height_tu)/4;
        real_t l = width_gl - 2*thickness_gl;
        real_t L = (y0_tu - y0_gl - thickness_gl) - (height_tu - height_gl)/2;
        
        real_t xphi_2;
        
        if ( sqrt(SQR(x-x0)) <= l/2 or sqrt(SQR(y-y0)) <= L/2 ){
            xphi_2 = FMAX(sqrt(SQR(x0-x))-l/2, sqrt(SQR(y0-y))-L/2);
        } else{
			if (y > y0 + L/2){
                if(x < x0 -l/2){
                    xphi_2 = sqrt(SQR(x - (x0-l/2)) + SQR(y - (y0+L/2)));
                }
                if(x > x0 + l/2){
                    xphi_2 = sqrt(SQR(x - (x0+l/2)) + SQR(y - (y0+L/2)));
                }
            }
            if (y < y0 - L/2){
                if(x < x0 -l/2){
                    xphi_2 = sqrt(SQR(x - (x0-l/2)) + SQR(y - (y0-L/2)));
                }
                if(x > x0 + l/2){
                    xphi_2 = sqrt(SQR(x - (x0+l/2)) + SQR(y - (y0-L/2)));
                }
            }
        }
        
        x0 = x0_gl + thickness_gl/2 - (width_gl + width_tu)/4;
        y0 = y0_gl - height_gl/2 + thickness_gl + h_glass/2;
        l = width_gl/2 - width_tu/2 - thickness_gl;
        L = h_glass;
        
        real_t xphi_3;
        
        if ( sqrt(SQR(x-x0)) <= l/2 or sqrt(SQR(y-y0)) <= L/2 ){
            xphi_3 = FMAX(sqrt(SQR(x0-x))-l/2, sqrt(SQR(y0-y))-L/2);
        }else{
            
            if (y > y0 + L/2){
                if(x < x0 -l/2){
                    xphi_3 = sqrt(SQR(x - (x0-l/2)) + SQR(y - (y0+L/2)));
                }
                if(x > x0 + l/2){
                    xphi_3 = sqrt(SQR(x - (x0+l/2)) + SQR(y - (y0+L/2)));
                }
            }
            if (y < y0 - L/2){
                if(x < x0 -l/2){
                    xphi_3 = sqrt(SQR(x - (x0-l/2)) + SQR(y - (y0-L/2)));
                }
                if(x > x0 + l/2){
                    xphi_3 = sqrt(SQR(x - (x0+l/2)) + SQR(y - (y0-L/2)));
                }
            }
        }
        
        x0 = x0_gl - thickness_gl/2 + (width_gl + width_tu)/4;
        y0 = y0_gl - height_gl/2 + thickness_gl + h_glass/2;
        l = width_gl/2 - width_tu/2 - thickness_gl;
        L = h_glass;
        
        real_t xphi_4;
        
        if ( sqrt(SQR(x-x0)) <= l/2 or sqrt(SQR(y-y0)) <= L/2 ){
            xphi_4 = FMAX(sqrt(SQR(x0-x))-l/2, sqrt(SQR(y0-y))-L/2);
        }else{
            
            if (y > y0 + L/2){
                if(x < x0 -l/2){
                    xphi_4 = sqrt(SQR(x - (x0-l/2)) + SQR(y - (y0+L/2)));
                }
                if(x > x0 + l/2){
                    xphi_4 = sqrt(SQR(x - (x0+l/2)) + SQR(y - (y0+L/2)));
                }
            }
            if (y < y0 - L/2){
                if(x < x0 -l/2){
                    xphi_4 = sqrt(SQR(x - (x0-l/2)) + SQR(y - (y0-L/2)));
                }
                if(x > x0 + l/2){
                    xphi_4 = sqrt(SQR(x - (x0+l/2)) + SQR(y - (y0-L/2)));
                }
            }
        }
        
        x0 = x0_gl;
        y0 = (y0_tu + y0_gl + thickness_gl - thickness_tu)/2 + (height_tu - height_gl)/4;
        l = width_tu - 2* thickness_tu;
        L = y0_tu - y0_gl - thickness_tu - thickness_gl + (height_tu + height_gl)/2;
        
        real_t xphi_5;
        
        if ( sqrt(SQR(x-x0)) <= l/2 or sqrt(SQR(y-y0)) <= L/2 ){
            xphi_5 = FMAX(sqrt(SQR(x0-x))-l/2, sqrt(SQR(y0-y))-L/2);
        }else{
            
            if (y > y0 + L/2){
                if(x < x0 -l/2){
                    xphi_5 = sqrt(SQR(x - (x0-l/2)) + SQR(y - (y0+L/2)));
                }
                if(x > x0 + l/2){
                    xphi_5 = sqrt(SQR(x - (x0+l/2)) + SQR(y - (y0+L/2)));
                }
            }
            if (y < y0 - L/2){
                if(x < x0 -l/2){
                    xphi_5 = sqrt(SQR(x - (x0-l/2)) + SQR(y - (y0-L/2)));
                }
                if(x > x0 + l/2){
                    xphi_5 = sqrt(SQR(x - (x0+l/2)) + SQR(y - (y0-L/2)));
                }
            }
        }
    
    xphi = FMIN(FMIN(FMIN(xphi_2,xphi_3),xphi_4),xphi_5);    
        
    }

    // init AC
    phi = Model.phi0(xphi);
    
    if (Model.initType == TWO_PHASE_ZALESAK) {
		if (x >= Model.xC-Model.R/6 &&
			x <= Model.xC+Model.R/6 &&
			y >= Model.yC-1.1*Model.R &&
			y <= Model.yC) phi = 0.0 ;
	}
	
	if (Model.initType == TWO_PHASE_SPINODAL_DECOMP) {
      phi = rand_gen.drand();
    }

    this->set_lbm_val(IJK, IPHI, phi);
    this->set_lbm_val(IJK, IC, c);
    this->set_lbm_val(IJK, IMU, mu);
	// ****************************************************************************************************
	//
	//                         INITIALIZATION OF VELOCITY IN THE LIQUID PHASE PHI
	//
	// ****************************************************************************************************
	if (Model.initType == TWO_PLANE) {
      //real_t L = Model.wave_length;
      vx = 0.5 * (Model.initVX_upper + Model.initVX_lower +
                  (Model.initVX_upper - Model.initVX_lower) *
                      tanh(Model.sign * 2.0 * xphi / Model.W));
    }
    else if (Model.initType == TWO_PHASE_SERPENTINE) {
      if (dim == 2) {
        vx = - pi*Model.U0 * cos(pi*((x/Model.L)-0.5)) * sin(pi*((y/Model.L)-0.5));
        vy =   pi*Model.U0 * sin(pi*((x/Model.L)-0.5)) * cos(pi*((y/Model.L)-0.5));
      }
    }
    
    else if (Model.initType == TWO_PHASE_ZALESAK) {
      if (dim == 2) {
        vx = Model.U0 * (2*y-1  );
        vy = Model.U0 * (1.0-2*x);
      }
    }
    
	else {
		vx = phi*Model.Vx_init ;
		vy = phi*Model.Vy_init ;
		vz = phi*Model.Vz_init ;
	}
	
	// ****************************************************************************************************
	//
	//                                 INITIALIZATION OF SOLID PHASE (PSI)
	//
	// ****************************************************************************************************
	if (Model.initTypeSol == SOLID_INIT_SPHERE) {
		xpsi = Model.psi_init_sphere(x, y, z) ;
	}
	else if (Model.initTypeSol == SOLID_INIT_CYLINDER){
        xpsi = Model.psi_init_cylinder3D(x, y, z) ;
	}
	else if (Model.initTypeSol == SOLID_INIT_CYLINDER_TANK){
		xpsi = Model.psi_init_cylinder_tank3D(x, y, z) ;
	}
	else if (Model.initTypeSol == SOLID_INIT_RECTANGLE){
		if (dim == 2) xpsi = Model.psi_init_Rectangle2D(x, y) ;
		else if (dim == 3) xpsi = Model.psi_init_Rectangle3D(x, y, z) ;
	}
	else if(Model.initTypeSol == SOLID_INIT_GLASS) {
		real_t x0_gl = Model.x0_gl;
		real_t y0_gl = Model.y0_gl;
		real_t height_gl = Model.height_gl;
		real_t width_gl = Model.width_gl;
		real_t thickness_gl = Model.thickness_gl;
	
		real_t xpsi_1;
		real_t x1 = x0_gl;
		real_t y1 = y0_gl - height_gl/2 + thickness_gl /2;
		real_t L = thickness_gl;
		real_t l = width_gl;
		if ( sqrt(SQR(x-x1)) <= l/2 or sqrt(SQR(y-y1)) <= L/2 ){
			xpsi_1 = FMIN(l/2 - sqrt(SQR(x1-x)),L/2 - sqrt(SQR(y1-y)));
		}else{
			
			if (y > y1 + L/2){
				if(x < x1 -l/2){
					xpsi_1 = -sqrt(SQR(x - (x1-l/2)) + SQR(y - (y1+L/2)));
				}
				if(x > x1 + l/2){
					xpsi_1 = -sqrt(SQR(x - (x1+l/2)) + SQR(y - (y1+L/2)));
				}
			}
			if (y < y1 - L/2){
				if(x < x1 -l/2){
					xpsi_1 = -sqrt(SQR(x - (x1-l/2)) + SQR(y - (y1-L/2)));
				}
				if(x > x1 + l/2){
					xpsi_1 = -sqrt(SQR(x - (x1+l/2)) + SQR(y - (y1-L/2)));
				}
			}
		}	
		
		
		real_t xpsi_2;
		real_t x2 = x0_gl - width_gl /2 + thickness_gl/2;
		real_t y2 = y0_gl;
		L = height_gl;
		l = thickness_gl;
	
		if ( sqrt(SQR(x-x2)) <= l/2 or sqrt(SQR(y-y2)) <= L/2 ){
			xpsi_2 = FMIN(l/2 - sqrt(SQR(x2-x)),L/2 - sqrt(SQR(y2-y)));
		} else{
			
			if (y > y2 + L/2){
				if(x < x2 -l/2){
					xpsi_2 = -sqrt(SQR(x - (x2-l/2)) + SQR(y - (y2+L/2)));
				}
				if(x > x2 + l/2){
					xpsi_2 = -sqrt(SQR(x - (x2+l/2)) + SQR(y - (y2+L/2)));
				}
			}
			if (y < y2 - L/2){
				if(x < x2 -l/2){
					xpsi_2 = -sqrt(SQR(x - (x2-l/2)) + SQR(y - (y2-L/2)));
				}
				if(x > x2 + l/2){
					xpsi_2 = -sqrt(SQR(x - (x2+l/2)) + SQR(y - (y2-L/2)));
				}
			}
		}
		
		real_t xpsi_3;
		real_t x3 = x0_gl + width_gl /2 - thickness_gl/2;
		real_t y3 = y0_gl;
		L = height_gl;
		l = thickness_gl;
	
		if ( sqrt(SQR(x-x3)) <= l/2 or sqrt(SQR(y-y3)) <= L/2 ){
			xpsi_3 = FMIN(l/2 - sqrt(SQR(x3-x)),L/2 - sqrt(SQR(y3-y)));
		} else{
			
			if (y > y3 + L/2){
				if(x < x3 -l/2){
					xpsi_3 = -sqrt(SQR(x - (x3-l/2)) + SQR(y - (y3+L/2)));
				}
				if(x > x3 + l/2){
					xpsi_3 = -sqrt(SQR(x - (x3+l/2)) + SQR(y - (y3+L/2)));
				}
			}
			if (y < y3 - L/2){
				if(x < x3 -l/2){
					xpsi_3 = -sqrt(SQR(x - (x3-l/2)) + SQR(y - (y3-L/2)));
				}
				if(x > x3 + l/2){
					xpsi_3 = -sqrt(SQR(x - (x3+l/2)) + SQR(y - (y3-L/2)));
				}
			}
		}
		
		xpsi = FMAX(FMAX(xpsi_1,xpsi_2),xpsi_3);
	}
	
	else if(Model.initTypeSol == SOLID_INIT_GLASS_HOLE){
		real_t x0_gl = Model.x0_gl;
		real_t y0_gl = Model.y0_gl;
		real_t height_gl = Model.height_gl;
		real_t width_gl = Model.width_gl;
		real_t thickness_gl = Model.thickness_gl;
		real_t lg_h = Model.length_hole_gl;
		real_t xpsi_1;
		real_t x1 = x0_gl - (lg_h + width_gl) /4 ;
		real_t y1 = y0_gl - height_gl/2 + thickness_gl /2;
		real_t L = thickness_gl;
		real_t l = width_gl/2 - lg_h/2;
	
		if ( sqrt(SQR(x-x1)) <= l/2 or sqrt(SQR(y-y1)) <= L/2 ){
			xpsi_1 = FMIN(l/2 - sqrt(SQR(x1-x)),L/2 - sqrt(SQR(y1-y)));
		}else{
			
			if (y > y1 + L/2){
				if(x < x1 -l/2){
					xpsi_1 = -sqrt(SQR(x - (x1-l/2)) + SQR(y - (y1+L/2)));
				}
				if(x > x1 + l/2){
					xpsi_1 = -sqrt(SQR(x - (x1+l/2)) + SQR(y - (y1+L/2)));
				}
			}
			if (y < y1 - L/2){
				if(x < x1 -l/2){
					xpsi_1 = -sqrt(SQR(x - (x1-l/2)) + SQR(y - (y1-L/2)));
				}
				if(x > x1 + l/2){
					xpsi_1 = -sqrt(SQR(x - (x1+l/2)) + SQR(y - (y1-L/2)));
				}
			}
		}	
		
		real_t xpsi_1_prime;
		x1 = x0_gl + (lg_h + width_gl) /4 ;
		y1 = y0_gl - height_gl/2 + thickness_gl /2;
		L = thickness_gl;
		l = width_gl/2 - lg_h/2;
	
		if ( sqrt(SQR(x-x1)) <= l/2 or sqrt(SQR(y-y1)) <= L/2 ){
			xpsi_1_prime = FMIN(l/2 - sqrt(SQR(x1-x)),L/2 - sqrt(SQR(y1-y)));
		}else{
			
			if (y > y1 + L/2){
				if(x < x1 -l/2){
					xpsi_1_prime = -sqrt(SQR(x - (x1-l/2)) + SQR(y - (y1+L/2)));
				}
				if(x > x1 + l/2){
					xpsi_1_prime = -sqrt(SQR(x - (x1+l/2)) + SQR(y - (y1+L/2)));
				}
			}
			if (y < y1 - L/2){
				if(x < x1 -l/2){
					xpsi_1_prime = -sqrt(SQR(x - (x1-l/2)) + SQR(y - (y1-L/2)));
				}
				if(x > x1 + l/2){
					xpsi_1_prime = -sqrt(SQR(x - (x1+l/2)) + SQR(y - (y1-L/2)));
				}
			}
		}
		
		real_t xpsi_2;
		real_t x2 = x0_gl - width_gl /2 + thickness_gl/2;
		real_t y2 = y0_gl;
		L = height_gl;
		l = thickness_gl;
	
		if ( sqrt(SQR(x-x2)) <= l/2 or sqrt(SQR(y-y2)) <= L/2 ){
			xpsi_2 = FMIN(l/2 - sqrt(SQR(x2-x)),L/2 - sqrt(SQR(y2-y)));
		}else{
			if (y > y2 + L/2){
				if(x < x2 -l/2){
					xpsi_2 = -sqrt(SQR(x - (x2-l/2)) + SQR(y - (y2+L/2)));
				}
				if(x > x2 + l/2){
					xpsi_2 = -sqrt(SQR(x - (x2+l/2)) + SQR(y - (y2+L/2)));
				}
			}
			if (y < y2 - L/2){
				if(x < x2 -l/2){
					xpsi_2 = -sqrt(SQR(x - (x2-l/2)) + SQR(y - (y2-L/2)));
				}
				if(x > x2 + l/2){
					xpsi_2 = -sqrt(SQR(x - (x2+l/2)) + SQR(y - (y2-L/2)));
				}
			}
		}
		
		real_t xpsi_3;
		real_t x3 = x0_gl + width_gl /2 - thickness_gl/2;
		real_t y3 = y0_gl;
		L = height_gl;
		l = thickness_gl;
	
		if ( sqrt(SQR(x-x3)) <= l/2 or sqrt(SQR(y-y3)) <= L/2 ){
			xpsi_3 = FMIN(l/2 - sqrt(SQR(x3-x)),L/2 - sqrt(SQR(y3-y)));
		}else{
			
			if (y > y3 + L/2){
				if(x < x3 -l/2){
					xpsi_3 = -sqrt(SQR(x - (x3-l/2)) + SQR(y - (y3+L/2)));
				}
				if(x > x3 + l/2){
					xpsi_3 = -sqrt(SQR(x - (x3+l/2)) + SQR(y - (y3+L/2)));
				}
			}
			if (y < y3 - L/2){
				if(x < x3 -l/2){
					xpsi_3 = -sqrt(SQR(x - (x3-l/2)) + SQR(y - (y3-L/2)));
				}
				if(x > x3 + l/2){
					xpsi_3 = -sqrt(SQR(x - (x3+l/2)) + SQR(y - (y3-L/2)));
				}
			}
		}
		
		xpsi = FMAX(FMAX(FMAX(xpsi_1, xpsi_1_prime),xpsi_2),xpsi_3);
	}
	else if(Model.initTypeSol == SOLID_INIT_TORRICELLI){
		real_t x0_gl = Model.x0_gl;
		real_t y0_gl = Model.y0_gl;
		real_t height_gl = Model.height_gl;
		real_t width_gl = Model.width_gl;
		real_t thickness_gl = Model.thickness_gl;
		real_t xpsi_1;
		real_t x1 = x0_gl;
		real_t y1 = y0_gl - height_gl/2 + thickness_gl /2;
		real_t L = thickness_gl;
		real_t l = width_gl;
		if ( sqrt(SQR(x-x1)) <= l/2 or sqrt(SQR(y-y1)) <= L/2 ){
			xpsi_1 = FMIN(l/2 - sqrt(SQR(x1-x)),L/2 - sqrt(SQR(y1-y)));
        } else{  
            if (y > y1 + L/2){
                if(x < x1 -l/2){
                    xpsi_1 = -sqrt(SQR(x - (x1-l/2)) + SQR(y - (y1+L/2)));
                }
                if(x > x1 + l/2){
                    xpsi_1 = -sqrt(SQR(x - (x1+l/2)) + SQR(y - (y1+L/2)));
                }
            }
            if (y < y1 - L/2){
                if(x < x1 -l/2){
                    xpsi_1 = -sqrt(SQR(x - (x1-l/2)) + SQR(y - (y1-L/2)));
                }
                if(x > x1 + l/2){
                    xpsi_1 = -sqrt(SQR(x - (x1+l/2)) + SQR(y - (y1-L/2)));
                }
            }
        }    
        
		real_t xpsi_2;
		real_t x2 = x0_gl - width_gl /2 + thickness_gl/2;
		real_t y2 = y0_gl;
		L = height_gl;
		l = thickness_gl;
    
		if ( sqrt(SQR(x-x2)) <= l/2 or sqrt(SQR(y-y2)) <= L/2 ){
            xpsi_2 = FMIN(l/2 - sqrt(SQR(x2-x)),L/2 - sqrt(SQR(y2-y)));
        } else{  
            if (y > y2 + L/2){
                if(x < x2 -l/2){
                    xpsi_2 = -sqrt(SQR(x - (x2-l/2)) + SQR(y - (y2+L/2)));
                }
                if(x > x2 + l/2){
                    xpsi_2 = -sqrt(SQR(x - (x2+l/2)) + SQR(y - (y2+L/2)));
                }
            }
            if (y < y2 - L/2){
                if(x < x2 -l/2){
                    xpsi_2 = -sqrt(SQR(x - (x2-l/2)) + SQR(y - (y2-L/2)));
                }
                if(x > x2 + l/2){
                    xpsi_2 = -sqrt(SQR(x - (x2+l/2)) + SQR(y - (y2-L/2)));
                }
            }
        }
        
		real_t xpsi_3;
		real_t x3 = x0_gl + width_gl /2 - thickness_gl/2;
		real_t y3 = y0_gl;
		L = height_gl;
		l = thickness_gl;
		if ( sqrt(SQR(x-x3)) <= l/2 or sqrt(SQR(y-y3)) <= L/2 ){
            xpsi_3 = FMIN(l/2 - sqrt(SQR(x3-x)),L/2 - sqrt(SQR(y3-y)));
        }else{
            
            if (y > y3 + L/2){
                if(x < x3 -l/2){
                    xpsi_3 = -sqrt(SQR(x - (x3-l/2)) + SQR(y - (y3+L/2)));
                }
                if(x > x3 + l/2){
                    xpsi_3 = -sqrt(SQR(x - (x3+l/2)) + SQR(y - (y3+L/2)));
                }
            }
            if (y < y3 - L/2){
                if(x < x3 -l/2){
                    xpsi_3 = -sqrt(SQR(x - (x3-l/2)) + SQR(y - (y3-L/2)));
                }
                if(x > x3 + l/2){
                    xpsi_3 = -sqrt(SQR(x - (x3+l/2)) + SQR(y - (y3-L/2)));
                }
            }
        }
        
    //  Tube
    
		real_t x0_tu = Model.x0_gl;
		real_t y0_tu = Model.y0_tu;
		real_t height_tu = Model.height_tu;
		real_t width_tu = Model.width_tu;
		real_t thickness_tu = Model.thickness_tu;
        
		real_t xpsi_4;
		x1 = x0_tu;
		y1 = y0_tu + height_tu/2 - thickness_tu /2;
		L = thickness_tu;
		l = width_tu;
    
		if ( sqrt(SQR(x-x1)) <= l/2 or sqrt(SQR(y-y1)) <= L/2 ){
            xpsi_4 = FMIN(l/2 - sqrt(SQR(x1-x)),L/2 - sqrt(SQR(y1-y)));
        }else{
            
            if (y > y1 + L/2){
                if(x < x1 -l/2){
                    xpsi_4 = -sqrt(SQR(x - (x1-l/2)) + SQR(y - (y1+L/2)));
                }
                if(x > x1 + l/2){
                    xpsi_4 = -sqrt(SQR(x - (x1+l/2)) + SQR(y - (y1+L/2)));
                }
            }
            if (y < y1 - L/2){
                if(x < x1 -l/2){
                    xpsi_4 = -sqrt(SQR(x - (x1-l/2)) + SQR(y - (y1-L/2)));
                }
                if(x > x1 + l/2){
                    xpsi_4 = -sqrt(SQR(x - (x1+l/2)) + SQR(y - (y1-L/2)));
                }
            }
        }    
        
        
		real_t xpsi_5;
		x2 = x0_tu - width_tu /2 + thickness_tu/2;
		y2 = y0_tu;
		L = height_tu;
		l = thickness_tu;
    
		if ( sqrt(SQR(x-x2)) <= l/2 or sqrt(SQR(y-y2)) <= L/2 ){
            xpsi_5 = FMIN(l/2 - sqrt(SQR(x2-x)),L/2 - sqrt(SQR(y2-y)));
        }else{
            
            if (y > y2 + L/2){
                if(x < x2 -l/2){
                    xpsi_5 = -sqrt(SQR(x - (x2-l/2)) + SQR(y - (y2+L/2)));
                }
                if(x > x2 + l/2){
                    xpsi_5 = -sqrt(SQR(x - (x2+l/2)) + SQR(y - (y2+L/2)));
                }
            }
            if (y < y2 - L/2){
                if(x < x2 -l/2){
                    xpsi_5 = -sqrt(SQR(x - (x2-l/2)) + SQR(y - (y2-L/2)));
                }
                if(x > x2 + l/2){
                    xpsi_5 = -sqrt(SQR(x - (x2+l/2)) + SQR(y - (y2-L/2)));
                }
            }
        }
        
		real_t xpsi_6;
		x3 = x0_tu + width_tu /2 - thickness_tu/2;
		y3 = y0_tu;
		L = height_tu;
		l = thickness_tu;
    
		if ( sqrt(SQR(x-x3)) <= l/2 or sqrt(SQR(y-y3)) <= L/2 ){
            xpsi_6 = FMIN(l/2 - sqrt(SQR(x3-x)),L/2 - sqrt(SQR(y3-y)));
        }else{
            
            if (y > y3 + L/2){
                if(x < x3 -l/2){
                    xpsi_6 = -sqrt(SQR(x - (x3-l/2)) + SQR(y - (y3+L/2)));
                }
                if(x > x3 + l/2){
                    xpsi_6 = -sqrt(SQR(x - (x3+l/2)) + SQR(y - (y3+L/2)));
                }
            }
            if (y < y3 - L/2){
                if(x < x3 -l/2){
                    xpsi_6 = -sqrt(SQR(x - (x3-l/2)) + SQR(y - (y3-L/2)));
                }
                if(x > x3 + l/2){
                    xpsi_6 = -sqrt(SQR(x - (x3+l/2)) + SQR(y - (y3-L/2)));
                }
            }
        }    
        
		xpsi = FMAX(FMAX(FMAX(FMAX(FMAX(xpsi_1,xpsi_2),xpsi_3),xpsi_4),xpsi_5),xpsi_6);
    }
	
	if (Model.solid_phase == 1) {
		psi = Model.psi0(xpsi) ;
	} else {
		psi = 0.0 ;
	}
	
	// Initialization of phi2
	real_t phi2 = 1.0 - phi - psi ;
	// Initialization of pressure p
	p = Model.p_init_phi * phi + Model.p_init_phi2 * phi2 ;
	
	// Initialization of source term for injector
	if (Model.initType == PHASE_FIELD_INIT_CYLINDER) {
		if (Model.injector == 1) Sinject = phi ;
	}
    // init NS
    //rho = Model.interp_rho(phi, c);
    
    rho = Model.interp_rho_with_phi_psi (phi,psi,phi2) ;
    
    // init composition c and chemical potential mu
    //c  = Model.c0(xc);
    c  = Model.c_co (phi,phi2,psi);
    mu = Model.mu(c, phi, phi2, psi);
    
    this->set_lbm_val(IJK, IPSI , psi    );
    this->set_lbm_val(IJK, IPHI2, phi2   );
    this->set_lbm_val(IJK, ISPHI, Sinject);
    this->set_lbm_val(IJK, IP   , p      );
    this->set_lbm_val(IJK, ID   , rho    );
    this->set_lbm_val(IJK, IU   , vx     );
    this->set_lbm_val(IJK, IV   , vy     );
    this->set_lbm_val(IJK, IC   , c      );
    if (dim == 3)
      this->set_lbm_val(IJK, IW, vz);

  } // end init macro

  // =====================================================================
  //                            UPDATE MACRO
  // =====================================================================

  KOKKOS_INLINE_FUNCTION
  void update_macro(IVect<dim> IJK) const {
    // get useful params
    const real_t dx  = this->params.dx;
    const real_t dt  = this->params.dt;
    const real_t cs2 = SQR(dx / dt) / Model.e2;
    
    if (dim==2) {
		LBMState lbmStatePrev;
		Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmStatePrev);
		RVect<dim> Solid_Vel ;
		real_t UBX        = 0.0 ;
		real_t UBY        = 0.0 ;
		// Solid velocity
		if (Model.initMoveSolid == SOLID_MOVE_OSCILL) {
			Solid_Vel = Model.Oscill_solid_velocity<dim>(lbmStatePrev);
		}
		else if (Model.initMoveSolid == SOLID_MOVE_ROTATE) {
			real_t x, y ;
			this->get_coordinates(IJK, x, y);
			Solid_Vel = Model.Rotate_solid_velocity<dim>(x, y, lbmStatePrev);
		}
		else {
			Solid_Vel [IX] = 0.0 ;
			Solid_Vel [IY] = 0.0 ;
		}
		UBX = Solid_Vel[IX] ;
		UBY = Solid_Vel[IY] ;
		int dbb = Model.diffuse_bounce_back;
		if (dbb == 1) {
			FState f_values;
			FState zeta;
			FState eta;
			real_t factr = 4.0*lbmStatePrev[IPSI] / Model.W_sol ;
			for (int ipop = 0; ipop < npop; ++ipop) {
				f_values[ipop] = Base::get_f_val(tagNS,IJK,ipop);
				real_t norm_grad_psi = sqrt( SQR(lbmStatePrev[IDPSIDX]) + SQR(lbmStatePrev[IDPSIDY]) ) ;
				if ( norm_grad_psi != 0.0) {
					real_t n_x = lbmStatePrev[IDPSIDX] / norm_grad_psi;
					real_t n_y = lbmStatePrev[IDPSIDY] / norm_grad_psi;
					zeta[ipop] = factr * FMAX( Base::compute_scal(ipop, n_x, n_y), 0.0) ;
				}
				else{
					zeta[ipop] = 0.0;
				}
				eta[ipop] = 2*w[ipop] * lbmStatePrev[ID] * Base::compute_scal(ipop, UBX, UBY) / cs2;
			}
		
			for (int ipop = 0; ipop < npop; ++ipop) {
				real_t c1 = Model.dt * zeta[ipop];
				real_t c2 = Model.dt * zeta[Ebar[ipop]];
				real_t fval = f_values[ipop] +(c1/(1+c1+c2))*(-f_values[ipop] + f_values[Ebar[ipop]] + eta[ipop] );
				this -> set_f_val( tagNS, IJK, ipop, fval);
				//this -> set_f_val( tagPHI, IJK, ipop, gval);
			}
		}
		
		// compute moments of distribution equations
		real_t moment_phi = 0.0;
		real_t moment_psi = 0.0;
		real_t moment_P   = 0.0;
		real_t moment_VX  = 0.0;
		real_t moment_VY  = 0.0;
		real_t moment_c   = 0.0;
	
		for (int ipop = 0; ipop < npop; ++ipop) {
			moment_phi += Base::get_f_val(tagPHI, IJK, ipop);
			moment_psi += Base::get_f_val(tagPSI, IJK, ipop);
			moment_P   += Base::get_f_val(tagNS, IJK, ipop);
			moment_VX  += Base::get_f_val(tagNS, IJK, ipop) * E[ipop][IX];
			moment_VY  += Base::get_f_val(tagNS, IJK, ipop) * E[ipop][IY];
			moment_c   += Base::get_f_val(tagCOMP, IJK, ipop);
		}

		// store old values of macro fields
		// const real_t rhoprev =lbmStatePrev[ID];
		const real_t ForceNSX = lbmStatePrev[IFX];
		const real_t ForceNSY = lbmStatePrev[IFY];

		// get source term for PHI
		const real_t Gst_phi  = Model.S_st(tagPHI, lbmStatePrev);
		const real_t Gdw_phi  = Model.S_dw(tagPHI, lbmStatePrev);
		// get source term for PSI
		const real_t Gst_psi  = Model.St_psi (tagPSI, lbmStatePrev);
		const real_t Gdw_psi  = Model.Sdw_psi(tagPSI, lbmStatePrev);
		// get source term for C
		const real_t Gst_comp = Model.St_C (lbmStatePrev);
		// compute new macro fields phi, psi, c
		const real_t phi      = moment_phi + 0.5 * dt * (Gst_phi + Gdw_phi);
		const real_t psi      = moment_psi + 0.5 * dt * (Gst_psi + Gdw_psi);
		const real_t c        = moment_c   + 0.5 * dt * (Gst_comp         );
		// Update time derivative of psi
		real_t dpsidt = (psi - lbmStatePrev[IPSI])/dt;
		
		// When psi does not follow a PDE
		//const real_t psi  = Model.Update_psi_CT(lbmStatePrev);
		//const real_t psi  = Model.Update_psi_LaxWendroff(lbmStatePrev);
		
		// Update phi2
		const real_t phi2 = 1.0 - phi - psi ;
		const real_t mu   = Model.mu(c, phi, phi2, psi);
		//const real_t rho  = Model.interp_rho(phi, c);
		const real_t rho  = Model.interp_rho_with_phi_psi(phi,psi,phi2) ;
		// compute NS macro vars
		real_t corrSt_P = Model.Sterm_solid_velocity_CT (lbmStatePrev) ;
		const real_t P  = (moment_P - 0.5*dt*corrSt_P) * (rho*cs2) ;
		real_t VX = moment_VX + 0.5 * dt * ForceNSX / rho;
		real_t VY = moment_VY + 0.5 * dt * ForceNSY / rho;
		if (Model.initType == TWO_PHASE_SERPENTINE || Model.initType == TWO_PHASE_ZALESAK) {
			VX = lbmStatePrev[IU];
			VY = lbmStatePrev[IV];
		}

		// update macro fields
		this->set_lbm_val(IJK, IPHI   , phi   );
		this->set_lbm_val(IJK, IPHI2  , phi2  );
		this->set_lbm_val(IJK, IPSI   , psi   );
		this->set_lbm_val(IJK, ID     , rho   );
		this->set_lbm_val(IJK, IP     , P     );
		this->set_lbm_val(IJK, IU     , VX    );
		this->set_lbm_val(IJK, IV     , VY    );
		this->set_lbm_val(IJK, IDPSIDT, dpsidt);
		this->set_lbm_val(IJK, IUB_X  , UBX   );
		this->set_lbm_val(IJK, IUB_Y  , UBY   );
		this->set_lbm_val(IJK, IC     , c     );
		this->set_lbm_val(IJK, IMU    , mu    );
	}
	
	if (dim==3) {
		// compute moments of distribution equations
		real_t moment_phi = 0.0;
		real_t moment_P   = 0.0;
		real_t moment_VX  = 0.0;
		real_t moment_VY  = 0.0;
		real_t moment_VZ  = 0.0;
		real_t moment_c   = 0.0;
	
		for (int ipop = 0; ipop < npop; ++ipop) {
			moment_phi += Base::get_f_val(tagPHI, IJK, ipop);
			moment_P += Base::get_f_val(tagNS, IJK, ipop);
			moment_VX += Base::get_f_val(tagNS, IJK, ipop) * E[ipop][IX];
			moment_VY += Base::get_f_val(tagNS, IJK, ipop) * E[ipop][IY];
			moment_VZ += Base::get_f_val(tagNS, IJK, ipop) * E[ipop][IZ];
			moment_c += Base::get_f_val(tagCOMP, IJK, ipop);
			}

    // store old values of macro fields
		LBMState lbmStatePrev;
		Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmStatePrev);

    // const real_t rhoprev =lbmStatePrev[ID];
		const real_t ForceNSX = lbmStatePrev[IFX];
		const real_t ForceNSY = lbmStatePrev[IFY];
		const real_t ForceNSZ = lbmStatePrev[IFZ];

    // get source term
		const real_t G_st = Model.S_st(tagPHI, lbmStatePrev);
		const real_t G_dw = Model.S_dw(tagPHI, lbmStatePrev);

    // compute new macro fields
		const real_t phi = moment_phi + 0.5 * dt * (G_st + G_dw);
		const real_t c   = moment_c;
		
		//const real_t rho = Model.interp_rho(phi, c);
		const real_t psi = lbmStatePrev[IPSI];
		if (Model.move_solid == 1) {
			const real_t psi  = Model.Update_psi_CT(lbmStatePrev);
			//const real_t psi  = Model.Update_psi_LaxWendroff(lbmStatePrev);
		}
		const real_t phi2 = 1.0 - phi - psi ;
		const real_t rho  = Model.interp_rho_with_phi_psi(phi,psi,phi2) ;
		const real_t mu  = Model.mu(c, phi, phi2, psi);
    // compute NS macro vars
		const real_t P = moment_P * rho * cs2;
		const real_t VX = moment_VX + 0.5 * dt * ForceNSX / rho;
		const real_t VY = moment_VY + 0.5 * dt * ForceNSY / rho;
		const real_t VZ = moment_VZ + 0.5 * dt * ForceNSZ / rho;
		
		real_t S_inject = 0.0 ;
		if (Model.injector == 1) S_inject = Model.Update_S_injection(lbmStatePrev);

    // update macro fields
		this->set_lbm_val(IJK, IPHI , phi     );
		this->set_lbm_val(IJK, IPHI2, phi2    );
		this->set_lbm_val(IJK, IPSI , psi     );
		this->set_lbm_val(IJK, ISPHI, S_inject);
		this->set_lbm_val(IJK, ID   , rho     );
		this->set_lbm_val(IJK, IP   , P       );
		this->set_lbm_val(IJK, IU   , VX      );
		this->set_lbm_val(IJK, IV   , VY      );
		this->set_lbm_val(IJK, IW   , VZ      );
		this->set_lbm_val(IJK, IC   , c       );
		this->set_lbm_val(IJK, IMU  , mu      );
	}

  } // end update macro

  // =====================================================================
  //                            UPDATE MACRO GRAD
  // =====================================================================

  KOKKOS_INLINE_FUNCTION
  void update_macro_grad(IVect<dim> IJK) const {
    RVect<dim> gradPhi;
    this->compute_gradient(gradPhi, IJK, IPHI, BOUNDARY_EQUATION_1);
    this->set_lbm_val(IJK, IDPHIDX, gradPhi[IX]);
    this->set_lbm_val(IJK, IDPHIDY, gradPhi[IY]);
    if (dim == 3)
      this->set_lbm_val(IJK, IDPHIDZ, gradPhi[IZ]);
      
    RVect<dim> gradPhi2;
    this->compute_gradient(gradPhi2, IJK, IPHI2, BOUNDARY_EQUATION_1);
    this->set_lbm_val(IJK, IDPHI2DX, gradPhi2[IX]);
    this->set_lbm_val(IJK, IDPHI2DY, gradPhi2[IY]);
    if (dim == 3)
      this->set_lbm_val(IJK, IDPHI2DZ, gradPhi2[IZ]);  
      
    RVect<dim> gradPsi;
    this->compute_gradient(gradPsi, IJK, IPSI, BOUNDARY_EQUATION_1);
    this->set_lbm_val(IJK, IDPSIDX, gradPsi[IX]);
    this->set_lbm_val(IJK, IDPSIDY, gradPsi[IY]);
    if (dim == 3)
      this->set_lbm_val(IJK, IDPSIDZ, gradPsi[IZ]);  
    
    real_t laplaPhi = Base::compute_laplacian(IJK, IPHI, BOUNDARY_EQUATION_1);
    Base::set_lbm_val(IJK, ILAPLAPHI, laplaPhi);
 // ATTENTION EQUATION_1 en ARGUMENT)
	real_t laplaPhi2 = 0.0 ;
	real_t laplaPsi  = 0.0 ;
	if (Model.contact_angle == 1) {
		laplaPhi2 = Base::compute_laplacian(IJK, IPHI2, BOUNDARY_EQUATION_1);
		laplaPsi  = Base::compute_laplacian(IJK, IPSI , BOUNDARY_EQUATION_1);
	}
	Base::set_lbm_val(IJK, ILAPLAPHI2, laplaPhi2);
	Base::set_lbm_val(IJK, ILAPLAPSI, laplaPsi);

    RVect<dim> gradRho;
    this->compute_gradient(gradRho, IJK, ID, BOUNDARY_EQUATION_2);
    this->set_lbm_val(IJK, IDRHODX, gradRho[IX]);
    this->set_lbm_val(IJK, IDRHODY, gradRho[IY]);
    if (dim == 3)
      this->set_lbm_val(IJK, IDRHODZ, gradRho[IZ]);

    RVect<dim> gradC;
    this->compute_gradient(gradC, IJK, IC, BOUNDARY_EQUATION_3);
    this->set_lbm_val(IJK, IDCDX, gradC[IX]);
    this->set_lbm_val(IJK, IDCDY, gradC[IY]);
    if (dim == 3)
      this->set_lbm_val(IJK, IDCDZ, gradC[IZ]);
  }

}; // end class

} // namespace PBM_NS_AC_Compo
#endif
