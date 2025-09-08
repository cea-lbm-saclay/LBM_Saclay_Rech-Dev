#ifndef LBMSCHEME_NS_PF_F_H_
#define LBMSCHEME_NS_PF_F_H_

#include <limits> // for std::numeric_limits

#ifdef __CUDA_ARCH__
#include <math_constants.h>
#endif // __CUDA_ARCH__

#include "FieldManager.h"
#include "LBMParams.h"
#include "LBM_Base_Functor.h"

#include "Models_NS_AC_Fakhari.h"

namespace PBM_NS_AC_Fakhari {
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

    ModelParams Model;
    modelType type;
    LBM_speeds E;
    LBM_speeds_opposite Ebar;
    LBM_Weights w;
    EquationTag1 tagPHI;
    EquationTag2 tagNS;

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

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag1 tag, const IVect<dim>& IJK, BGKCollider<dim, npop>& collider) const
    {

        // Paramètres pour la simulation
        const real_t dt = Model.dt;
        const real_t dx = Model.dx;
        const real_t e2 = Model.e2;
        const real_t cs2 = 1.0 / e2 * SQR(dx / dt);

        // Stockage des anciennes grandeurs macroscopiques
        LBMState lbmState;
        Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmState);

        const real_t M0 = Model.M0_PHI(tag, lbmState);
        const RVect2 M1 = Model.M1_PHI(tag, lbmState);
        const real_t M2 = Model.M2_PHI(tag, lbmState);

        const real_t G_dw = Model.S_dw(tag, lbmState);
        const RVect2 G_ct = Model.S_ct(tag, lbmState);

        // Calcul du tau de collision
        collider.tau = Model.tau_PHI(tag, lbmState);
        real_t staudx = 1 / ((collider.tau - 0.5) * dx / e2);

        // ipop = 0
        collider.f[0] = Base::get_f_val(tag, IJK, 0);
        collider.S0[0] = w[0] * dt * (G_dw + staudx * Base::compute_scal(0, G_ct));
        collider.feq[0] = M0 - (1 - w[0]) * M2 - 0.5 * collider.S0[0];

        // ipop > 0
        for (int ipop = 1; ipop < npop; ++ipop) {
            collider.f[ipop] = this->get_f_val(tag, IJK, ipop);
            collider.S0[ipop] = w[ipop] * dt * (G_dw + staudx * Base::compute_scal(ipop, G_ct));
            collider.feq[ipop] = w[ipop] * (M2 + dx / dt * Base::compute_scal(ipop, M1) / cs2) - 0.5 * (collider.S0[ipop]);
        }
    } // end of setup_collider for phase field equation

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag1 tag, const IVect<dim>& IJK, MRTCollider<dim, npop>& collider) const
    {
    }

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag2 tag, const IVect<dim>& IJK, BGKCollider<dim, npop>& collider) const
    {

        // Paramètres pour la simulation
        const real_t dx = Model.dx;
        const real_t dt = Model.dt;
        const real_t cs2 = SQR(dx / dt) / Model.e2;
        const real_t c = dx / dt;

        // Stockage des anciennes grandeurs macroscopiques
        LBMState lbmState;
        Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmState);

        // Calcul du tau de collision
        collider.tau = Model.tau_NS(type, lbmState);

        // Calcul de l'équilibre sans terme source
        FState GAMMA;
        real_t scalUU = SQR(lbmState[IU]) + SQR(lbmState[IV]);
        for (int ipop = 0; ipop < npop; ++ipop) {
            real_t scalUC = c * Base::compute_scal(ipop, lbmState[IU], lbmState[IV]);
            GAMMA[ipop] = scalUC / cs2 + 0.5 * SQR(scalUC) / SQR(cs2) - 0.5 * scalUU / cs2;
            real_t feqbar = w[ipop] * (lbmState[IP] / (cs2 * lbmState[ID]) + GAMMA[ipop]);
            collider.feq[ipop] = feqbar;
            collider.f[ipop] = Base::get_f_val(tag, IJK, ipop);
        }

        // Calculs des termes forces pour le terme source

        RVect2 ForceTS = Model.force_TS(type, lbmState);
        RVect2 ForceG = Model.force_G(type, lbmState);
        RVect2 ForceP = Model.force_P(type, lbmState);
        RVect2 ForceV;
        RVect2 ForceTot;

        const real_t mu = Model.nu0 * Model.rho0 + lbmState[IPHI] * (Model.nu1 * Model.rho1 - Model.nu0 * Model.rho0);
        const real_t nu = mu / lbmState[ID];
        real_t coeffV = -(nu * (Model.rho1 - Model.rho0)) / (cs2 * dt * collider.tau) * SQR(c);

        ForceV[IX] = 0.0;
        ForceV[IY] = 0.0;

        for (int alpha = 0; alpha < npop; ++alpha) {
            ForceV[IX] += E[alpha][IX] * E[alpha][IX] * lbmState[IDPHIDX] * (collider.f[alpha] - collider.feq[alpha]) + E[alpha][IX] * E[alpha][IY] * lbmState[IDPHIDY] * (collider.f[alpha] - collider.feq[alpha]);
            ForceV[IY] += E[alpha][IY] * E[alpha][IX] * lbmState[IDPHIDX] * (collider.f[alpha] - collider.feq[alpha]) + E[alpha][IY] * E[alpha][IY] * lbmState[IDPHIDY] * (collider.f[alpha] - collider.feq[alpha]);
        }

        ForceV[IX] = coeffV * ForceV[IX];
        ForceV[IY] = coeffV * ForceV[IY];

        // Calcul et enregistrement du terme source total
        ForceTot[IX] = ForceG[IX] + ForceP[IX] + ForceTS[IX] + ForceV[IX];
        this->set_lbm_val(IJK, IFX, ForceTot[IX]);
        ForceTot[IY] = ForceG[IY] + ForceP[IY] + ForceTS[IY] + ForceV[IY];
        this->set_lbm_val(IJK, IFY, ForceTot[IY]);

        // Rajout de la source dans l'équilibre
        for (int ipop = 0; ipop < npop; ++ipop) {
            collider.S0[ipop] = dx * w[ipop] / (lbmState[ID] * cs2) * Base::compute_scal(ipop, ForceTot[IX], ForceTot[IY]);
            ;
            collider.feq[ipop] = collider.feq[ipop] - 0.5 * (collider.S0[ipop]);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag2 tag, const IVect<dim>& IJK, MRTCollider<dim, npop>& collider) const
    {

        // Setup matrices MRT
        //~ collider.setMandMinv(Base::lattice.M, Base::lattice.Minv);

        // Paramètres pour la simulation
        const real_t dx = Model.dx;
        //const real_t dtprev = Model.dtprev;
        const real_t dt = Model.dt;
        const real_t c = dx / dt;
        const real_t cs2 = SQR(c) / Model.e2;

        // Stockage des anciennes grandeurs macroscopiques
        LBMState lbmState;
        Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmState);

        // compute collision rate
        collider.tau = Model.tau_NS(type, lbmState);

        for (int i = 0; i < npop; i++) {
            for (int j = 0; j < npop; j++) {
                if (i != j) {
                    collider.S[i][j] = 0.0;
                } else if (Model.tauMatrixNS[i] != -1.0) {
                    collider.S[i][j] = Model.tauMatrixNS[i];
                } else {
                    collider.S[i][j] = 1 / collider.tau;
                }
            }
        }

        // Calcul de l'équilibre sans terme source
        FState GAMMA;
        real_t scalUU = SQR(lbmState[IU]) + SQR(lbmState[IV]);

        for (int ipop = 0; ipop < npop; ++ipop) {

            real_t scalUC = c * Base::compute_scal(ipop, lbmState[IU], lbmState[IV]);
            GAMMA[ipop] = scalUC / cs2 + 0.5 * SQR(scalUC) / SQR(cs2) - 0.5 * scalUU / cs2;
            real_t feqbar = w[ipop] * (lbmState[IP] / (cs2 * lbmState[ID]) + GAMMA[ipop]);
            collider.feq[ipop] = feqbar;
            collider.f[ipop] = Base::get_f_val(tag, IJK, ipop);
        }

        // Calculs des termes forces pour le terme source
        RVect2 ForceTS = Model.force_TS(type, lbmState);
        RVect2 ForceG = Model.force_G(type, lbmState);
        RVect2 ForceP = Model.force_P(type, lbmState);
        RVect2 ForceV;
        RVect2 ForceTot;

        const real_t mu = Model.nu0 * Model.rho0 + lbmState[IPHI] * (Model.nu1 * Model.rho1 - Model.nu0 * Model.rho0);
        const real_t nu = mu / lbmState[ID];
        real_t coeffV = -(nu * (Model.rho1 - Model.rho0)) / (cs2 * dt) * SQR(c);
        collider.Calc_Lf();
        FState MinvSMg_geq = collider.Prod_invMSMf;

        ForceV[IX] = 0.0;
        ForceV[IY] = 0.0;

        for (int beta = 0; beta < npop; ++beta) {
            ForceV[IX] += E[beta][IX] * E[beta][IX] * lbmState[IDPHIDX] * MinvSMg_geq[beta] + E[beta][IX] * E[beta][IY] * lbmState[IDPHIDY] * MinvSMg_geq[beta];
            ForceV[IY] += E[beta][IY] * E[beta][IX] * lbmState[IDPHIDX] * MinvSMg_geq[beta] + E[beta][IY] * E[beta][IY] * lbmState[IDPHIDY] * MinvSMg_geq[beta];
        }

        ForceV[IX] = coeffV * ForceV[IX];
        ForceV[IY] = coeffV * ForceV[IY];

        //real_t v_dt = -(dt-dtprev)/(2*dtprev);
        //RVect2 Force_VDT;
        //Force_VDT[IX] = lbmState[ID]*lbmState[IU]*v_dt;
        //Force_VDT[IY] = lbmState[ID]*lbmState[IV]*v_dt;

        ForceTot[IX] = ForceG[IX] + ForceP[IX] + ForceTS[IX] + ForceV[IX]; // + Force_VDT[IX];
        this->set_lbm_val(IJK, IFX, ForceTot[IX]);
        ForceTot[IY] = ForceG[IY] + ForceP[IY] + ForceTS[IY] + ForceV[IY]; // + Force_VDT[IY];
        this->set_lbm_val(IJK, IFY, ForceTot[IY]);

        // Rajout de la source dans l'équilibre
        for (int ipop = 0; ipop < npop; ++ipop) {
            collider.S0[ipop] = dx * w[ipop] / (lbmState[ID] * cs2) * Base::compute_scal(ipop, ForceTot[IX], ForceTot[IY]);
            ;
            collider.feq[ipop] = collider.feq[ipop] - 0.5 * (collider.S0[ipop]);
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
    };

    // =====================================================================
    //							INIT MACRO
    // =====================================================================

    KOKKOS_INLINE_FUNCTION
    void init_macro(IVect<dim> IJK, RANDOM_POOL::generator_type rand_gen) const
    {

        // get local coordinates
        real_t x, y;
        this->get_coordinates(IJK, x, y);

        // get useful params
        real_t xx = 0.0;
        real_t phi = 0.0;

        real_t vx = 0.0;
        real_t vy = 0.0;

        // compute phase field
        if (Model.initType == PHASE_FIELD_INIT_VERTICAL)
            xx = x - Model.x0;
        else if (Model.initType == PHASE_FIELD_INIT_SPHERE)
            xx = (Model.r0 - sqrt(SQR(x - Model.x0) + SQR(y - Model.y0)));
        else if (Model.initType == PHASE_FIELD_INIT_2SPHERE) {
            real_t d = sqrt(SQR(x - Model.x0) + SQR(y - Model.y0));
            bool ri = (d > ((Model.r0 + Model.r1) / 2));
            xx = ri * (Model.r0 - d) - (1 - ri) * (Model.r1 - d);
        } else if (Model.initType == PHASE_FIELD_INIT_SQUARE)
            xx = FMIN(Model.r0 - sqrt(SQR(x - Model.x0)), Model.r1 - sqrt(SQR(y - Model.y0)));
        else if (Model.initType == PHASE_FIELD_INIT_MIXTURE)
            xx = x - Model.x0;
        else if (Model.initType == PHASE_FIELD_INIT_TAYLOR) {
            real_t L = Model.wave_length;
            xx = y - Model.ampl * sin(Model.n_wave * 2.0 * 3.14 / L * x + 3.14 / 2.0) - Model.height;
        }

        if (Model.initType == PHASE_FIELD_INIT_DATA)
            phi = this->get_lbm_val(IJK, IPHI);
        else
            phi = Model.phi0(xx);

        // set macro fields values
        if (!(Model.initType == PHASE_FIELD_INIT_DATA))
            this->set_lbm_val(IJK, IPHI, phi);

        // init NS
        this->set_lbm_val(IJK, IP, 0.0);
        this->set_lbm_val(IJK, ID, Model.rho0);
        this->set_lbm_val(IJK, IU, vx);
        this->set_lbm_val(IJK, IV, vy);

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
        const real_t c = dx / dt;

        // compute moments of distribution equations
        real_t moment_phi = 0.0;
        real_t moment_P = 0.0;
        real_t moment_VX = 0.0;
        real_t moment_VY = 0.0;
        for (int ipop = 0; ipop < npop; ++ipop) {
            moment_phi += Base::get_f_val(tagPHI, IJK, ipop);
            moment_P += Base::get_f_val(tagNS, IJK, ipop);
            moment_VX += Base::get_f_val(tagNS, IJK, ipop) * E[ipop][IX] * c;
            moment_VY += Base::get_f_val(tagNS, IJK, ipop) * E[ipop][IY] * c;
        }

        // store old values of macro fields
        LBMState lbmStatePrev;
        Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmStatePrev);

        const real_t rhoprev = lbmStatePrev[ID];
        const real_t ForceNSX = lbmStatePrev[IFX];
        const real_t ForceNSY = lbmStatePrev[IFY];

        // get source term
        const real_t G_dw = Model.S_dw(tagPHI, lbmStatePrev);

        // compute new macro fields
        const real_t phi = moment_phi + 0.5 * dt * G_dw;
        const real_t rho = Model.interp_rho(phi);
        // compute NS macro vars
        const real_t P = moment_P * rhoprev * cs2;
        const real_t VX = moment_VX + 0.5 * dt * ForceNSX / rhoprev;
        const real_t VY = moment_VY + 0.5 * dt * ForceNSY / rhoprev;
        // update macro fields
        this->set_lbm_val(IJK, IPHI, phi);
        this->set_lbm_val(IJK, ID, rho);
        this->set_lbm_val(IJK, IP, P);
        this->set_lbm_val(IJK, IU, VX);
        this->set_lbm_val(IJK, IV, VY);
        if (dim == 3)
            this->set_lbm_val(IJK, IV, VY);

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
        if (dim == 3) {
            Base::set_lbm_val(IJK, IDPHIDZ, gradPhi[IZ]);
        }
        real_t laplaPhi = Base::compute_laplacian(IJK, IPHI, BOUNDARY_EQUATION_1);
        Base::set_lbm_val(IJK, ILAPLAPHI, laplaPhi);
    }

}; // end class

} // end namespace
#endif
