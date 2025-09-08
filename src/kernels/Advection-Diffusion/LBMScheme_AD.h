#ifndef LBMSCHEME_AD_H_
#define LBMSCHEME_AD_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>

#endif // __CUDA_ARCH__

#include "FieldManager.h"
#include "LBMParams.h"
#include "LBM_Base_Functor.h"

#include "Models_AD.h"

namespace PBM_AD {
template <int dim, int npop, typename modelType>
struct LBMScheme : public LBMSchemeBase<dim, npop> {

    //! Vérifier ce qui est utile

    // USING
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
    EquationTag1 tagC;

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

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag1 tag, const IVect<dim>& IJK, BGK_Collider& collider) const
    {

        const real_t dt = Model.dt;
        const real_t dtprev = Model.dtprev;
        const real_t dx = Model.dx;
        const real_t cs = 1.0 / sqrt(3) * dx / dt;
        const real_t cs2 = pow(cs, 2);

        LBMState lbmState;
        Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmState);

        const real_t M0 = Model.M0(type, tag, lbmState);
        const RVect<dim> M1 = Model.M1<dim>(type, tag, lbmState);

        // compute collision rate
        collider.tau = Model.tau(type, tag, lbmState);

        // compute correction due to variation of timestep
        RVect<dim> S_corr_dt = M1;
        real_t c = (collider.tau - 3 / 2) * dt * 1.0 / dtprev * log(dtprev / dt);
        S_corr_dt[IX] = c * M1[IX];
        S_corr_dt[IY] = c * M1[IY];

        // Compute feq
        for (int ipop = 0; ipop < npop; ++ipop) {
            collider.f[ipop] = this->get_f_val(tag, IJK, ipop);

            collider.S0[ipop] = dt * w[ipop] * Base::compute_scal(ipop, S_corr_dt) * dx / dt;

            real_t cia_ua = Base::compute_scal(ipop, lbmState[IU], lbmState[IV]) * dx / dt;

            collider.feq[ipop] = w[ipop] * M0 * (1 + cia_ua / cs2) - 0.5 * (collider.S0[ipop]);
        }
    } // end of setup_collider for phase field equation

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag1 tag, const IVect<dim>& IJK, MRT_Collider& collider) const
    {

        const real_t dt = Model.dt;
        const real_t dx = Model.dx;

        LBMState lbmState;
        Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmState);

        const real_t M0 = Model.M0(type, tag, lbmState);
        const real_t M2 = Model.M2(type, tag, lbmState);

        collider.tau = Model.tau(type, tag, lbmState);

        // compute collision rate

        for (int i = 0; i < npop; i++) {
            for (int j = 0; j < npop; j++) {
                if (i != j) {
                    collider.S[i][j] = 0.0;
                } else if (Model.tauMatrix[i] != -1.0) {
                    collider.S[i][j] = Model.tauMatrix[i];
                } else {
                    collider.S[i][j] = 1 / collider.tau;
                }
            }
        }

        // ipop = 0
        collider.f[0] = Base::get_f_val(tag, IJK, 0);
        collider.S0[0] = 0.0;
        collider.feq[0] = M0 - (1 - w[0]) * M2 * (1 + 3 * dt * Base::compute_scal(0, lbmState[IU], lbmState[IV]) / dx) - 0.5 * collider.S0[0];

        // ipop > 0
        for (int ipop = 1; ipop < npop; ++ipop) {
            collider.f[ipop] = this->get_f_val(tag, IJK, ipop);
            collider.S0[ipop] = 0.0;
            collider.feq[ipop] = w[ipop] * M2 * (1 + 3 * dt * Base::compute_scal(ipop, lbmState[IU], lbmState[IV]) / dx) - 0.5 * (collider.S0[ipop]);
        }
    } // end of setup_collider for phase field equation

    //==================================================================

    KOKKOS_INLINE_FUNCTION
    void make_boundary(const IVect<dim>& IJK, int faceId) const
    {

        // phi boundary
        if (Base::params.boundary_types[BOUNDARY_EQUATION_1][faceId] == BC_ANTI_BOUNCE_BACK) {
            real_t boundary_value = Base::params.boundary_values[BOUNDARY_PHASE_FIELD][faceId];

            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_antibounceback(tagC, faceId, IJK, ipop, boundary_value);

        }

        else if (Base::params.boundary_types[BOUNDARY_EQUATION_1][faceId] == BC_ZERO_FLUX) {
            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_bounceback(tagC, faceId, IJK, ipop, 0.0);
        }

        //! Pas de boundarie périodic pour phi ??
    };

    //==================================================================

    //! Ici encore tout en 2D
    KOKKOS_INLINE_FUNCTION
    void init_macro(IVect<dim> IJK, RANDOM_POOL::generator_type rand_gen) const
    {

        // get local coordinates
        real_t x, y;
        this->get_coordinates(IJK, x, y);

        real_t c = 0.0;
        real_t vx = 0.0;
        real_t vy = 0.0;

        if (Model.initType == AD_1D_INIT_TUBE) {
            real_t t = fmod(Model.time, Model.period);
            real_t x0 = Model.x0;
            c = Model.C0 * exp(-(x - x0) * (x - x0) / (2 * Model.r1));
            vx = Model.U0 * (exp(-(t - Model.t1) * (t - Model.t1)) - exp(-(t - Model.t2) * (t - Model.t2)));
        }

        this->set_lbm_val(IJK, IC, c);
        this->set_lbm_val(IJK, IU, vx);
        this->set_lbm_val(IJK, IV, vy);
        this->set_lbm_val(IJK, IT, Model.time);

    } // end init macro

    KOKKOS_INLINE_FUNCTION
    void update_macro(IVect<dim> IJK) const
    {
        // get local coordinates
        real_t x, y;
        this->get_coordinates(IJK, x, y);

        // compute moments of distribution equations
        real_t moment_c = 0.0;

        for (int ipop = 0; ipop < npop; ++ipop) {
            moment_c += Base::get_f_val(tagC, IJK, ipop);
        }

        // store old values of macro fields
        LBMState lbmStatePrev;
        Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmStatePrev);

        // get source terms and compute new macro fields
        const real_t c = moment_c;

        // update macro fields
        this->set_lbm_val(IJK, IC, c);
        this->set_lbm_val(IJK, IT, Model.time);

        // update field ini

        if (Model.initType == AD_1D_INIT_TUBE) {
            real_t vx = 0.0;
            real_t t = fmod(Model.time, Model.period);
            vx = Model.U0 * (exp(-(t - Model.t1) * (t - Model.t1)) - exp(-(t - Model.t2) * (t - Model.t2)));
            this->set_lbm_val(IJK, IU, vx);
        }

    } // end update macro

    KOKKOS_INLINE_FUNCTION
    void update_macro_grad(IVect<dim> IJK) const
    {
    }

}; // end class LBMScheme_AD

} // end namespace
#endif
