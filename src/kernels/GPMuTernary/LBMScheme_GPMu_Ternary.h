#ifndef LBMSCHEME_GPMU_TERNARY_H_
#define LBMSCHEME_GPMU_TERNARY_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>

#endif // __CUDA_ARCH__

#include "FieldManager.h"
#include "LBMParams.h"
#include "LBM_Base_Functor.h"

#include "Models_GP_Mu_Ternary.h"
namespace PBM_GP_MU_TERNARY {
template <int dim, int npop, typename modelType>
struct LBMScheme : public LBMSchemeBase<dim, npop> {

    using LBMState = typename Kokkos::Array<real_t, COMPONENT_SIZE>;
    using id2index_t = Kokkos::Array<int, COMPONENT_SIZE>;
    using LBMArray = typename LBMBaseFunctor<dim, npop>::LBMArray;
    using FArray = typename LBMBaseFunctor<dim, npop>::FArray;
    using LBM_Weights = typename LBMBaseFunctor<dim, npop>::LBM_Weights;
    using LBM_speeds = typename LBMBaseFunctor<dim, npop>::LBM_speeds;
    using FState = typename Kokkos::Array<real_t, npop>;
    //~ using namespace localDataTypes;

    static constexpr real_t NORMAL_EPSILON = 1.0e-16;

    LBMArray lbm_data;
    ModelParams Model;
    modelType type;
    LBM_Weights w;
    EquationTag1 tagPHI;
    EquationTag2 tagC;
    EquationTag3 tagCB;

    LBMScheme()
        : LBMSchemeBase<dim, npop>()
        , w(LBMBaseFunctor<dim, npop>::w) {};

    LBMScheme(LBMParams params, LBMArray& lbm_data, ModelParams& Model)
        : LBMSchemeBase<dim, npop>(params, lbm_data)
        , lbm_data(lbm_data)
        , Model(Model)
        , w(LBMBaseFunctor<dim, npop>::w) {};

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag1 tag, IVect<dim>& IJK, BGKCollider<npop>& collider) const
    {

        const real_t dt = this->params.dt;

        LBMState lbmState;
        this->setupLBMState(IJK, lbmState);

        const real_t source = Model.source_phi(type, lbmState);
        const RVect<dim> factor = Model.force_phi<dim>(type, lbmState);

        // compute collision rate
        collider.tau = Model.tau_phi(type, lbmState);

        // compute feq and FS
        for (int ipop = 0; ipop < npop; ++ipop) {
            collider.f[ipop] = this->get_f_val(tag, IJK, ipop);
            real_t scal = this->compute_scal(ipop, factor);

            collider.S0[ipop] = dt * w[ipop] * (source) + dt * w[ipop] * (scal);
            //~ collider.S1[ipop] = dt * w[ipop]* (scal);

            collider.feq[ipop] = w[ipop] * lbmState[IPHI] - 0.5 * (collider.S0[ipop]);
        }
    } // end of setup_collider for phase field equation

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag2 tag, IVect<dim>& IJK, BGKColliderTimeFactor<npop>& collider) const
    {

        const real_t dt = this->params.dt;

        IVect<dim> IJKs;
        bool can_str;
        LBMState lbmState;
        this->setupLBMState(IJK, lbmState);
        const real_t source = Model.source_A(type, lbmState);
        const RVect<dim> factor = Model.force_CA<dim>(type, lbmState);
        const real_t Mbar = Model.compute_mobilityA(type, lbmState);

        // compute collision rate
        collider.tau = Model.tau_CA(type, lbmState);
        collider.factor = Model.kiA(type, lbmState);
        // ipop = 0

        collider.f[0] = this->get_f_val(tag, IJK, 0);

        can_str = this->stream_alldir(IJK, IJKs, 0);
        if (can_str) {
            collider.f_nonlocal[0] = this->get_f_val(tag, IJKs, 0);
        } else {
            collider.f_nonlocal[0] = 0.0;
        }

        real_t scal = this->compute_scal(0, factor);
        collider.FAdvec[0] = dt * w[0] * (scal);
        collider.Source[0] = dt * w[0] * (source);
        collider.feq[0] = lbmState[IMU] - (lbmState[IMU] * Mbar + 100) * (1 - w[0]) - 0.5 * collider.FAdvec[0];

        // ipop > 0
        for (int ipop = 1; ipop < npop; ++ipop) {
            collider.f[ipop] = this->get_f_val(tag, IJK, ipop);

            can_str = this->stream_alldir(IJK, IJKs, ipop);
            if (can_str) {
                collider.f_nonlocal[ipop] = this->get_f_val(tag, IJKs, ipop);
            } else {
                collider.f_nonlocal[ipop] = 0.0;
            }

            scal = this->compute_scal(ipop, factor);
            collider.FAdvec[ipop] = dt * w[ipop] * (scal);
            collider.Source[ipop] = dt * w[ipop] * (source);
            collider.feq[ipop] = w[ipop] * (lbmState[IMU] * Mbar + 100) - 0.5 * collider.FAdvec[ipop];
        }
    } // end of setup_collider for composition equation

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag3 tag, IVect<dim>& IJK, BGKColliderTimeFactor<npop>& collider) const
    {

        const real_t dt = this->params.dt;

        IVect<dim> IJKs;
        bool can_str;
        LBMState lbmState;
        this->setupLBMState(IJK, lbmState);

        const real_t source = Model.source_B(type, lbmState);
        const RVect<dim> factor = Model.force_CB<dim>(type, lbmState);
        const real_t Mbar = Model.compute_mobilityB(type, lbmState);

        // compute collision rate
        collider.tau = Model.tau_CB(type, lbmState);
        collider.factor = Model.kiB(type, lbmState);
        // ipop = 0
        collider.f[0] = this->get_f_val(tag, IJK, 0);

        can_str = this->stream_alldir(IJK, IJKs, 0);
        if (can_str) {
            collider.f_nonlocal[0] = this->get_f_val(tag, IJKs, 0);
        } else {
            collider.f_nonlocal[0] = 0.0;
        }

        real_t scal = this->compute_scal(0, factor);
        collider.FAdvec[0] = dt * w[0] * (scal);
        collider.Source[0] = dt * w[0] * (source);
        collider.feq[0] = lbmState[IMUB] - (lbmState[IMUB] * Mbar) * (1 - w[0]) - 0.5 * collider.FAdvec[0];

        // ipop > 0
        for (int ipop = 1; ipop < npop; ++ipop) {
            collider.f[ipop] = this->get_f_val(tag, IJK, ipop);

            can_str = this->stream_alldir(IJK, IJKs, ipop);
            if (can_str) {
                collider.f_nonlocal[ipop] = this->get_f_val(tag, IJKs, ipop);
            } else {
                collider.f_nonlocal[ipop] = 0.0;
            }

            scal = this->compute_scal(ipop, factor);
            collider.FAdvec[ipop] = dt * w[ipop] * (scal);
            collider.Source[ipop] = dt * w[ipop] * (source);
            collider.feq[ipop] = w[ipop] * (lbmState[IMUB] * Mbar) - 0.5 * collider.FAdvec[ipop];
        }
    } // end of setup_collider for composition equation

    KOKKOS_INLINE_FUNCTION
    void make_boundary(IVect<dim>& IJK, int faceId) const
    {
        //~ LBMState lbmState;
        //~ this->setupLBMState(IJK, lbmState);
        // phi boundary
        if (this->params.boundary_types[BOUNDARY_EQUATION_1][faceId] == BC_ANTI_BOUNCE_BACK) {
            real_t boundary_value = this->params.boundary_values[BOUNDARY_PHASE_FIELD][faceId];

            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_antibounceback(tagPHI, faceId, IJK, ipop, boundary_value);

        }

        else if (this->params.boundary_types[BOUNDARY_EQUATION_1][faceId] == BC_ZERO_FLUX) {
            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_bounceback(tagPHI, faceId, IJK, ipop, 0.0);
        }

        // C A boundary
        if (this->params.boundary_types[BOUNDARY_EQUATION_2][faceId] == BC_ANTI_BOUNCE_BACK) {
            real_t boundary_value_C = this->params.boundary_values[BOUNDARY_CONCENTRATION][faceId];
            real_t boundary_value_phi = this->params.boundary_values[BOUNDARY_PHASE_FIELD][faceId];
            real_t boundary_value = Model.gammaA * (Model.compute_muA(type, boundary_value_phi, boundary_value_C)) * Model.DA1 * boundary_value_C;
            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_antibounceback(tagC, faceId, IJK, ipop, boundary_value);
        }

        else if (this->params.boundary_types[BOUNDARY_EQUATION_2][faceId] == BC_ZERO_FLUX) {
            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_bounceback(tagC, faceId, IJK, ipop, 0.0);
        }
        // C B boundary
        if (this->params.boundary_types[BOUNDARY_EQUATION_3][faceId] == BC_ANTI_BOUNCE_BACK) {
            real_t boundary_value_C = this->params.boundary_values[BOUNDARY_CONCENTRATION][faceId];
            real_t boundary_value_phi = this->params.boundary_values[BOUNDARY_PHASE_FIELD][faceId];
            real_t boundary_value = Model.gammaB * (Model.compute_muB(type, boundary_value_phi, boundary_value_C)) * Model.DB1 * boundary_value_C;
            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_antibounceback(tagCB, faceId, IJK, ipop, boundary_value);
        }

        else if (this->params.boundary_types[BOUNDARY_EQUATION_3][faceId] == BC_ZERO_FLUX) {
            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_bounceback(tagCB, faceId, IJK, ipop, 0.0);
        }
    };

    KOKKOS_INLINE_FUNCTION
    void init_macro(IVect<dim>& IJK, RANDOM_POOL::generator_type rand_gen) const
    {

        // get local coordinates
        real_t x, y;
        this->get_coordinates(IJK, x, y);

        real_t xx = 0.0;
        real_t phi = 0.0;
        // compute phase field
        if (Model.initType == PHASE_FIELD_INIT_VERTICAL or Model.initType == PHASE_FIELD_RANDOM_TERNARY_GLASS)
            xx = x - Model.x0;
        else if (Model.initType == PHASE_FIELD_INIT_SPHERE or Model.initType == PHASE_FIELD_RANDOM_TERNARY_GLASS_SPHERE)
            xx = (Model.r0 - sqrt(SQR(x - Model.x0) + SQR(y - Model.y0)));
        else if (Model.initType == PHASE_FIELD_INIT_2SPHERE) {
            real_t xx1 = (Model.r0 - sqrt(SQR(x - Model.x0) + SQR(y - Model.y0)));
            real_t xx2 = (Model.r2 - sqrt(SQR(x - Model.x2) + SQR(y - Model.y2)));
            xx = FMAX(xx1, xx2);
        } else if (Model.initType == PHASE_FIELD_INIT_SQUARE)
            xx = (Model.r0 - FMAX(sqrt(SQR(x - Model.x0)), sqrt(SQR(y - Model.y0))));
        else if (Model.initType == PHASE_FIELD_INIT_MIXTURE)
            xx = x - Model.x0;

        if (Model.initType == PHASE_FIELD_INIT_DATA)
            phi = this->get_lbm_val(IJK, IPHI);
        else
            phi = Model.phi0(xx);

        // get useful params
        const real_t initClA = Model.initClA;
        const real_t initCsA = Model.initCsA;
        real_t initClB = Model.initClB;
        real_t initCsB = Model.initCsB;
        if (Model.initType == PHASE_FIELD_RANDOM_TERNARY_GLASS or Model.initType == PHASE_FIELD_RANDOM_TERNARY_GLASS_SPHERE) {
            initCsB += 0.3 * (1 + cos(6.3 * Model.freq * IJK[IX] / this->params.nx) * cos(6.3 * Model.freq * IJK[IY] / this->params.ny));
        }

        real_t h = Model.h(phi);
        const real_t CA = (1.0 - h) * initCsA + h * initClA;
        const real_t CB = (1.0 - h) * initCsB + h * initClB;
        // compute mu
        const real_t muA = Model.compute_muA(type, phi, CA);
        const real_t muB = Model.compute_muB(type, phi, CB);

        // set macro fields values
        if (!(Model.initType == PHASE_FIELD_INIT_DATA))
            this->set_lbm_val(IJK, IPHI, phi);
        this->set_lbm_val(IJK, IC, CA);
        this->set_lbm_val(IJK, ICB, CB);
        this->set_lbm_val(IJK, IMU, muA);
        this->set_lbm_val(IJK, IMUB, muB);

        // init grad phi
        this->set_lbm_val(IJK, IDPHIDX, 0.0);
        this->set_lbm_val(IJK, IDPHIDY, 0.0);

        // init time derivatives
        if (IDPHIDT >= 0) {
            this->set_lbm_val(IJK, IDPHIDT, 0.0);
        }
        if (IDMUDT >= 0) {
            this->set_lbm_val(IJK, IDMUDT, 0.0);
        }

    } // end init macro

    KOKKOS_INLINE_FUNCTION
    void update_macro(IVect<dim>& IJK) const
    {
        // get useful params
        const real_t dt = this->params.dt;

        // store old values of macro fields
        LBMState lbmStatePrev;
        this->setupLBMState(IJK, lbmStatePrev);

        // compute moments of distribution equations

        real_t moment_phi = 0.0;
        real_t moment_mu = 0.0;
        real_t moment_muB = 0.0;
        for (int ipop = 0; ipop < npop; ++ipop) {
            moment_phi += this->get_f_val(tagPHI, IJK, ipop);
            moment_mu += this->get_f_val(tagC, IJK, ipop);
            moment_muB += this->get_f_val(tagCB, IJK, ipop);
        }

        // get source terms and compute new macro fields
        const real_t source_phi = Model.source_phi(type, lbmStatePrev);
        const real_t source_A = Model.source_A(type, lbmStatePrev);
        const real_t source_B = Model.source_B(type, lbmStatePrev);

        real_t phi = FMIN(FMAX(moment_phi + 0.5 * dt * source_phi, 0.0), 1.0);
        real_t mu = moment_mu + 0.5 * dt * source_A;
        real_t muB = moment_muB + 0.5 * dt * source_B;

        // compute new chemical potential
        const real_t C = Model.compute_CA(type, phi, mu);
        const real_t CB = Model.compute_CB(type, phi, muB);

        // update macro fields
        this->set_lbm_val(IJK, IPHI, phi);
        this->set_lbm_val(IJK, IC, C);
        this->set_lbm_val(IJK, IMU, mu);
        this->set_lbm_val(IJK, ICB, CB);
        this->set_lbm_val(IJK, IMUB, muB);

        // update time derivatives if needed
        if (IDPHIDT >= 0) {
            this->set_lbm_val(IJK, IDPHIDT, (phi - lbmStatePrev[IPHI]) / dt);
        }
        if (IDMUDT >= 0) {
            this->set_lbm_val(IJK, IDMUDT, (mu - lbmStatePrev[IMU]) / dt);
        }
    } // end update macro

    KOKKOS_INLINE_FUNCTION
    void update_macro_grad(IVect<dim>& IJK) const
    {
        RVect<dim> gradPhi;
        this->compute_gradient(gradPhi, IJK, IPHI, BOUNDARY_EQUATION_1);
        this->set_lbm_val(IJK, IDPHIDX, gradPhi[IX]);
        this->set_lbm_val(IJK, IDPHIDY, gradPhi[IY]);
        if (dim == 3) {
            this->set_lbm_val(IJK, IDPHIDZ, gradPhi[IZ]);
        }
        if (IDCADX >= 0) {
            RVect<dim> gradCA;
            this->compute_gradient(gradCA, IJK, IC, BOUNDARY_EQUATION_2);
            this->set_lbm_val(IJK, IDCADX, gradCA[IX]);
            this->set_lbm_val(IJK, IDCADY, gradCA[IY]);
            if (dim == 3) {
                this->set_lbm_val(IJK, IDCADZ, gradCA[IZ]);
            }
        }
        if (IDCBDX >= 0) {
            RVect<dim> gradCB;
            this->compute_gradient(gradCB, IJK, ICB, BOUNDARY_EQUATION_3);
            this->set_lbm_val(IJK, IDCBDX, gradCB[IX]);
            this->set_lbm_val(IJK, IDCBDY, gradCB[IY]);
            if (dim == 3) {
                this->set_lbm_val(IJK, IDCBDZ, gradCB[IZ]);
            }
        }
    }

}; // end class LBMScheme_ADE_std

}
#endif // KERNELS_FUNCTORS_H_
