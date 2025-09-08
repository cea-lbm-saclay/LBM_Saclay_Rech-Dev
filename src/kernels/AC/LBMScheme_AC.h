#ifndef LBMSCHEME_AC_H_
#define LBMSCHEME_AC_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>

#endif // __CUDA_ARCH__

#include "FieldManager.h"
#include "LBMParams.h"
#include "LBM_Base_Functor.h"

#include "Models_AC.h"

namespace PBM_AC {
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
    using TRT_Collider = TRTCollider<dim, npop>;
    using MRT_Collider = MRTCollider<dim, npop>;

    LBMArray lbm_data;
    ModelParams Model;
    modelType type;
    LBM_speeds E;
    LBM_speeds_opposite Ebar;
    LBM_Weights w;
    EquationTag1 tagPHI;

    LBMScheme()
        : LBMSchemeBase<dim, npop>()
        , w(LBMLattice<dim, npop>().w) {};

    LBMScheme(LBMParams& params, LBMArray& lbm_data, ModelParams& Model)
        : LBMSchemeBase<dim, npop>(params, lbm_data)
        , lbm_data(lbm_data)
        , Model(Model)
        , w(LBMLattice<dim, npop>().w) {};

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag1 tag, const IVect<dim>& IJK, BGK_Collider& collider) const
    {

        const real_t dt = Base::params.dt;
        const real_t dx = Base::params.dx;
        const real_t e2 = Model.e2;

        LBMState lbmState;
        Base::setupLBMState(IJK, lbmState);

        const real_t M0 = Model.M0(type, tag, lbmState);
        const real_t M2 = Model.M2(type, tag, lbmState);
        const real_t S0 = Model.S0(type, tag, lbmState);
        const RVect<dim> S1 = Model.S1<dim>(type, tag, lbmState);

        // compute collision rate
        collider.tau = Model.tau(type, tag, lbmState);
        real_t staudx = 1 / ((Model.tau(type, tag, lbmState) - 0.5) * dx / e2);

        // ipop = 0
        collider.f[0] = Base::get_f_val(tag, IJK, 0);
        real_t scal = Base::compute_scal(0, S1);
        collider.S0[0] = dt * w[0] * (S0 + staudx * scal);
        //~ collider.S1[0] = dt * w[0] * staudx * scal;
        collider.feq[0] = M0 - (1 - w[0]) * M2 - 0.5 * (collider.S0[0]);

        // ipop > 0
        for (int ipop = 1; ipop < npop; ++ipop) {
            collider.f[ipop] = Base::get_f_val(tag, IJK, ipop);
            scal = Base::compute_scal(ipop, S1);
            collider.S0[ipop] = dt * w[ipop] * (S0 + staudx * scal);
            //~ collider.S1[ipop] = dt * w[ipop] * staudx * scal;
            collider.feq[ipop] = w[ipop] * M2 - 0.5 * (collider.S0[ipop]);
        }
    } // end of setup_collider for phase field equation

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag1 tag, const IVect<dim>& IJK, TRT_Collider& collider) const
    {

        const real_t dt = Base::params.dt;
        const real_t dx = Base::params.dx;
        LBMState lbmState;
        Base::setupLBMState(IJK, lbmState);

        const real_t M0 = Model.M0(type, tag, lbmState);
        const real_t M2 = Model.M2(type, tag, lbmState);
        const real_t S0 = Model.S0(type, tag, lbmState);
        const RVect<dim> S1 = Model.S1<dim>(type, tag, lbmState);

        // compute collision rate
        collider.tauA = Model.tau(type, tag, lbmState);
        if (Base::params.TRT_tauMethod1 == FIXED_LAMBDA) {
            collider.tauS = 0.5 + Base::params.lambdaTRT1 / (collider.tauA - 0.5);
        } else if (Base::params.TRT_tauMethod1 == FIXED_TAU) {
            collider.tauS = Base::params.TRT_tauS1;
        } else if (Base::params.TRT_tauMethod1 == CONDITIONAL_TAU) {
            collider.tauS = FMAX(collider.tauA, 1.0);
        }
        real_t staudx = 1 / ((Model.tau(type, tag, lbmState) - 0.5) * dx / Model.e2);

        // ipop = 0
        collider.f[0] = Base::get_f_val(tag, IJK, 0);
        real_t scal = Base::compute_scal(0, S1);
        collider.S0[0] = dt * w[0] * (S0 + staudx * scal);
        //~ collider.S1[0] = dt * w[0] * staudx * scal;
        collider.feq[0] = M0 - (1 - w[0]) * M2 - 0.5 * collider.S0[0];

        // ipop > 0
        for (int ipop = 1; ipop < npop; ++ipop) {
            collider.f[ipop] = Base::get_f_val(tag, IJK, ipop);
            scal = Base::compute_scal(ipop, S1);
            collider.S0[ipop] = dt * w[ipop] * (S0 + staudx * scal);
            //~ collider.S1[ipop] = dt * w[ipop] * staudx * scal;
            collider.feq[ipop] = w[ipop] * M2 - 0.5 * (collider.S0[ipop]);
        }
    } // end of setup_collider for phase field equation

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag1 tag, const IVect<dim>& IJK, MRT_Collider& collider) const
    {

        const real_t dt = Model.dt;
        const real_t dx = Model.dx;
        const real_t e2 = Model.e2;

        LBMState lbmState;
        Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmState);

        const real_t M0 = Model.M0(type, tag, lbmState);
        const real_t M2 = Model.M2(type, tag, lbmState);
        const real_t S0 = Model.S0(type, tag, lbmState);
        const RVect<dim> S1 = Model.S1<dim>(type, tag, lbmState);

        collider.tau = Model.tau(type, tag, lbmState);
        real_t staudx = 1 / ((collider.tau - 0.5) * dx / e2);

        // compute collision rate
        for (int i = 0; i < npop; i++) {
            if (Model.tauMatrix[i] != -1.0) {
                collider.S[i] = Model.tauMatrix[i];
            } else {
                collider.S[i] = 1 / collider.tau;
            }
        }

        // ipop = 0
        collider.f[0] = Base::get_f_val(tag, IJK, 0);
        real_t scal = Base::compute_scal(0, S1);
        collider.S0[0] = dt * w[0] * (S0 + staudx * scal);
        collider.feq[0] = M0 - (1 - w[0]) * M2 * (1 + 3 * dt * Base::compute_scal(0, lbmState[IU], lbmState[IV]) / dx) - 0.5 * collider.S0[0];

        // ipop > 0
        for (int ipop = 1; ipop < npop; ++ipop) {
            collider.f[ipop] = this->get_f_val(tag, IJK, ipop);
            scal = Base::compute_scal(ipop, S1);
            collider.S0[ipop] = dt * w[ipop] * (S0 + staudx * scal);
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
                this->compute_boundary_antibounceback(tagPHI, faceId, IJK, ipop, boundary_value);

        }

        else if (Base::params.boundary_types[BOUNDARY_EQUATION_1][faceId] == BC_ZERO_FLUX) {
            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_bounceback(tagPHI, faceId, IJK, ipop, 0.0);
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

        real_t loc_interf = 0.0;
        real_t phi = 0.0;
        real_t vx = 0.0;
        real_t vy = 0.0;

        const real_t pi = M_PI;

        if (Model.initType == PHASE_FIELD_INIT_MIXTURE) {
            loc_interf = x - Model.x0; //! A MODIFIER
        } else if (Model.initType == PHASE_FIELD_INIT_SPHERE) {
            // init phase field
            loc_interf = (Model.r0 - sqrt(SQR(x - Model.x0) + SQR(y - Model.y0)));
            phi = Model.phi0(loc_interf);

            // init velocity
            vx = -Model.U0 * Model.time * cos(pi * (x - 0.5)) * sin(pi * (y - 0.5));
            vy = Model.U0 * Model.time * sin(pi * (x - 0.5)) * cos(pi * (y - 0.5));
        } else if (Model.initType == PHASE_FIELD_INIT_SHRF_VORTEX) {
            // init phase field
            loc_interf = (Model.r0 - sqrt(SQR(x - Model.x0) + SQR(y - Model.y0)));
            phi = Model.phi0(loc_interf);

            // init velocity
            vx = -Model.U0 * Model.time * cos(pi * (x - 0.5)) * sin(pi * (y - 0.5));
            vy = Model.U0 * Model.time * sin(pi * (x - 0.5)) * cos(pi * (y - 0.5));
        }

        if (Model.initType == PHASE_FIELD_INIT_DATA) {
            phi = this->get_lbm_val(IJK, IPHI);
        } else {
            this->set_lbm_val(IJK, IPHI, phi);
            this->set_lbm_val(IJK, IU, vx);
            this->set_lbm_val(IJK, IV, vy);
        }

    } // end init macro

    KOKKOS_INLINE_FUNCTION
    void update_macro(IVect<dim> IJK) const
    {
        // get useful params
        const real_t dt = this->params.dt;

        // compute moments of distribution equations
        real_t moment_phi = 0.0;

        for (int ipop = 0; ipop < npop; ++ipop) {
            moment_phi += Base::get_f_val(tagPHI, IJK, ipop);
        }

        // store old values of macro fields
        LBMState lbmStatePrev;
        Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmStatePrev);

        const real_t source_phi = Model.S0(type, tagPHI, lbmStatePrev);

        // get source terms and compute new macro fields
        const real_t phi = moment_phi + 0.5 * dt * source_phi;

        // update macro fields
        this->set_lbm_val(IJK, IPHI, phi);

        // update field ini

        if (Model.initType == PHASE_FIELD_INIT_SHRF_VORTEX) {
            real_t vx, vy;
            const real_t pi = M_PI;
            real_t x, y;
            this->get_coordinates(IJK, x, y);
            vx = -Model.U0 * Model.time * cos(pi * (x - 0.5)) * sin(pi * (y - 0.5));
            vy = Model.U0 * Model.time * sin(pi * (x - 0.5)) * cos(pi * (y - 0.5));
            this->set_lbm_val(IJK, IU, vx);
            this->set_lbm_val(IJK, IV, vy);
        }

    } // end update macro

    //! cette fonction sera surement à modifier

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

}; // end class LBMScheme_AC_std

} // end namespace
#endif // KERNELS_FUNCTORS_H_
