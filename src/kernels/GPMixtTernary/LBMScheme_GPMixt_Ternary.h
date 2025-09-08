/**
 * \file LBMScheme_GPMixt_Ternary.h
 *
 */

#ifndef LBMSCHEME_GPMIXT_TERNARY_H_
#define LBMSCHEME_GPMIXT_TERNARY_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>

#endif // __CUDA_ARCH__

#include "FieldManager.h"
#include "LBMParams.h"
#include "LBM_Base_Functor.h"
#include "kokkos_shared.h"

#include "Models_GP_Mixt_Ternary.h"
namespace PBM_GP_MIXT_TERNARY
{
template<int dim, int npop, typename modelType>
struct LBMScheme : public LBMSchemeBase<dim, npop>
{
    using Base = LBMSchemeBase<dim, npop>;

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
      , w(LBMLattice<dim, npop>().w){};

    LBMScheme(LBMParams& params, LBMArray& lbm_data, ModelParams& Model)
      : LBMSchemeBase<dim, npop>(params, lbm_data)
      , lbm_data(lbm_data)
      , Model(Model)
      , w(LBMLattice<dim, npop>().w)
      , CCid_vv(KOKKOS_MAP_DEFAULT_CAPACITY)
      , map_CCid_to_phi(KOKKOS_MAP_DEFAULT_CAPACITY)
      , map_CCid_to_phiClA(KOKKOS_MAP_DEFAULT_CAPACITY)
      , map_CCid_to_phiClB(KOKKOS_MAP_DEFAULT_CAPACITY)

      {};

    // =======================================================
    // ==== BGK ==============================================
    // =======================================================

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag1 tag, const IVect<dim>& IJK, BGKCollider<dim, npop>& collider) const
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
        for (int ipop = 1; ipop < npop; ++ipop)
        {
            collider.f[ipop] = Base::get_f_val(tag, IJK, ipop);
            scal = Base::compute_scal(ipop, S1);
            collider.S0[ipop] = dt * w[ipop] * (S0 + staudx * scal);
            //~ collider.S1[ipop] = dt * w[ipop] * staudx * scal;
            collider.feq[ipop] = w[ipop] * M2 - 0.5 * (collider.S0[ipop]);
        }
    } // end of setup_collider for phase field equation

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag2 tag, const IVect<dim>& IJK, BGKCollider<dim, npop>& collider) const
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
        for (int ipop = 1; ipop < npop; ++ipop)
        {
            collider.f[ipop] = Base::get_f_val(tag, IJK, ipop);
            scal = Base::compute_scal(ipop, S1);
            collider.S0[ipop] = dt * w[ipop] * (S0 + staudx * scal);
            //~ collider.S1[ipop] = dt * w[ipop] * staudx * scal;
            collider.feq[ipop] = w[ipop] * M2 - 0.5 * (collider.S0[ipop]);
        }
    } // end of setup_collider for phase field equation

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag3 tag, IVect<dim>& IJK, BGKCollider<dim, npop>& collider) const
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
        for (int ipop = 1; ipop < npop; ++ipop)
        {
            collider.f[ipop] = Base::get_f_val(tag, IJK, ipop);
            scal = Base::compute_scal(ipop, S1);
            collider.S0[ipop] = dt * w[ipop] * (S0 + staudx * scal);
            //~ collider.S1[ipop] = dt * w[ipop] * staudx * scal;
            collider.feq[ipop] = w[ipop] * M2 - 0.5 * (collider.S0[ipop]);
        }
    } // end of setup_collider for phase field equation

    // =======================================================
    // ==== TRT ==============================================
    // =======================================================

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag1 tag, const IVect<dim>& IJK, TRTCollider<dim, npop>& collider) const
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
        if (Base::params.TRT_tauMethod1 == FIXED_LAMBDA)
        {
            collider.tauS = 0.5 + Base::params.lambdaTRT1 / (collider.tauA - 0.5);
        }
        else if (Base::params.TRT_tauMethod1 == FIXED_TAU)
        {
            collider.tauS = Base::params.TRT_tauS1;
        }
        else if (Base::params.TRT_tauMethod1 == CONDITIONAL_TAU)
        {
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
        for (int ipop = 1; ipop < npop; ++ipop)
        {
            collider.f[ipop] = Base::get_f_val(tag, IJK, ipop);
            scal = Base::compute_scal(ipop, S1);
            collider.S0[ipop] = dt * w[ipop] * (S0 + staudx * scal);
            //~ collider.S1[ipop] = dt * w[ipop] * staudx * scal;
            collider.feq[ipop] = w[ipop] * M2 - 0.5 * (collider.S0[ipop]);
        }
    } // end of setup_collider for phase field equation

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag2 tag, const IVect<dim>& IJK, TRTCollider<dim, npop>& collider) const
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
        if (Base::params.TRT_tauMethod2 == FIXED_LAMBDA)
        {
            collider.tauS = 0.5 + Base::params.lambdaTRT2 / (collider.tauA - 0.5);
        }
        else if (Base::params.TRT_tauMethod2 == FIXED_TAU)
        {
            collider.tauS = Base::params.TRT_tauS2;
        }
        else if (Base::params.TRT_tauMethod2 == CONDITIONAL_TAU)
        {
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
        for (int ipop = 1; ipop < npop; ++ipop)
        {
            collider.f[ipop] = Base::get_f_val(tag, IJK, ipop);
            scal = Base::compute_scal(ipop, S1);
            collider.S0[ipop] = dt * w[ipop] * (S0 + staudx * scal);
            //~ collider.S1[ipop] = dt * w[ipop] * staudx * scal;
            collider.feq[ipop] = w[ipop] * M2 - 0.5 * (collider.S0[ipop]);
        }
    } // end of setup_collider for phase field equation

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag3 tag, const IVect<dim>& IJK, TRTCollider<dim, npop>& collider) const
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
        if (Base::params.TRT_tauMethod3 == FIXED_LAMBDA)
        {
            collider.tauS = 0.5 + Base::params.lambdaTRT3 / (collider.tauA - 0.5);
        }
        else if (Base::params.TRT_tauMethod3 == FIXED_TAU)
        {
            collider.tauS = Base::params.TRT_tauS3;
        }
        else if (Base::params.TRT_tauMethod3 == CONDITIONAL_TAU)
        {
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
        for (int ipop = 1; ipop < npop; ++ipop)
        {
            collider.f[ipop] = Base::get_f_val(tag, IJK, ipop);
            scal = Base::compute_scal(ipop, S1);
            collider.S0[ipop] = dt * w[ipop] * (S0 + staudx * scal);
            //~ collider.S1[ipop] = dt * w[ipop] * staudx * scal;
            collider.feq[ipop] = w[ipop] * M2 - 0.5 * (collider.S0[ipop]);
        }
    } // end of setup_collider for phase field equation

    KOKKOS_INLINE_FUNCTION
    void make_boundary(const IVect<dim>& IJK, int faceId) const
    {

        LBMState lbmStateNext;
        Base::setupLBMState(IJK, lbmStateNext);

        lbmStateNext[IPHI] = Base::params.boundary_values[BOUNDARY_PHASE_FIELD][faceId];
        lbmStateNext[IC] = Base::params.boundary_values[BOUNDARY_CONCENTRATION][faceId];
        lbmStateNext[IMU] = Model.compute_muA(type, lbmStateNext[IPHI], lbmStateNext[IC]);
        lbmStateNext[ICB] = Base::params.boundary_values[BOUNDARY_CONCENTRATIONB][faceId];
        lbmStateNext[IMUB] = Model.compute_muB(type, lbmStateNext[IPHI], lbmStateNext[ICB]);

        // phi boundary
        if (Base::params.boundary_types[BOUNDARY_EQUATION_1][faceId] == BC_ANTI_BOUNCE_BACK)
        {
            real_t boundary_value = Model.M2(type, tagPHI, lbmStateNext) - 0.5 * Model.dt * Model.S0(type, tagPHI, lbmStateNext);

            for (int ipop = 1; ipop < npop; ++ipop)
            {
                Base::compute_boundary_antibounceback(tagPHI, faceId, IJK, ipop, boundary_value);
            }
        }

        else if (Base::params.boundary_types[BOUNDARY_EQUATION_1][faceId] == BC_ZERO_FLUX)
        {
            for (int ipop = 1; ipop < npop; ++ipop)
            {
                Base::compute_boundary_bounceback(tagPHI, faceId, IJK, ipop, 0.0);
            }
        }

        // CA boundary
        if (Base::params.boundary_types[BOUNDARY_EQUATION_2][faceId] == BC_ANTI_BOUNCE_BACK)
        {
            real_t boundary_value = Model.M2(type, tagC, lbmStateNext) - 0.5 * Model.dt * Model.S0(type, tagC, lbmStateNext);
            for (int ipop = 1; ipop < npop; ++ipop)
            {
                Base::compute_boundary_antibounceback(tagC, faceId, IJK, ipop, boundary_value);
            }
        }

        else if (Base::params.boundary_types[BOUNDARY_EQUATION_2][faceId] == BC_ZERO_FLUX)
        {
            for (int ipop = 1; ipop < npop; ++ipop)
            {
                Base::compute_boundary_bounceback(tagC, faceId, IJK, ipop, 0.0);
            }
        }
        // CB boundary
        if (Base::params.boundary_types[BOUNDARY_EQUATION_3][faceId] == BC_ANTI_BOUNCE_BACK)
        {
            real_t boundary_value = Model.M2(type, tagCB, lbmStateNext) - 0.5 * Model.dt * Model.S0(type, tagCB, lbmStateNext);
            for (int ipop = 1; ipop < npop; ++ipop)
            {
                Base::compute_boundary_antibounceback(tagCB, faceId, IJK, ipop, boundary_value);
            }
        }

        else if (Base::params.boundary_types[BOUNDARY_EQUATION_3][faceId] == BC_ZERO_FLUX)
        {
            for (int ipop = 1; ipop < npop; ++ipop)
            {
                Base::compute_boundary_bounceback(tagCB, faceId, IJK, ipop, 0.0);
            }
        }
    };

    KOKKOS_INLINE_FUNCTION
    void init_macro(const IVect<dim>& IJK, RANDOM_POOL::generator_type rand_gen) const
    {

        // get local coordinates
        real_t x, y;
        Base::get_coordinates(IJK, x, y);

        real_t xx = 0.0;
        real_t phi = 0.0;
        // compute phase field
        if (Model.initType == PHASE_FIELD_INIT_VERTICAL or Model.initType == PHASE_FIELD_RANDOM_TERNARY_GLASS)
        {
            xx = x - Model.x0;
        }
        else if (Model.initType == PHASE_FIELD_INIT_VERTICAL_PERTURBED)
        {
            xx = x - Model.x0 + Model.r0 * cos(6.3 * Model.freq * IJK[IY] / Base::params.ny);
        }
        else if (Model.initType == PHASE_FIELD_INIT_VERTICAL_PERTURBED_RANDOM)
        {
            xx = x - Model.x0 + Model.r0 * (rand_gen.drand());
        }
        else if (Model.initType == PHASE_FIELD_INIT_SPHERE or Model.initType == PHASE_FIELD_RANDOM_TERNARY_GLASS_SPHERE)
        {
            xx = (Model.r0 - sqrt(SQR(x - Model.x0) + SQR(y - Model.y0)));
        }
        else if (Model.initType == PHASE_FIELD_INIT_2SPHERE)
        {
            real_t xx1 = (Model.r0 - sqrt(SQR(x - Model.x0) + SQR(y - Model.y0)));
            real_t xx2 = (Model.r2 - sqrt(SQR(x - Model.x2) + SQR(y - Model.y2)));
            xx = FMAX(xx1, xx2);
        }
        else if (Model.initType == PHASE_FIELD_INIT_SQUARE)
        {
            xx = (Model.r0 - FMAX(sqrt(SQR(x - Model.x0)), sqrt(SQR(y - Model.y0))));
        }
        else if (Model.initType == PHASE_FIELD_INIT_MIXTURE)
        {
            xx = x - Model.x0;
        }
        else if (Model.initType == PHASE_FIELD_INIT_VERTICAL_ERFC_1INT)
        {
            xx = x - Model.xi * SQRT(Model.init_time);
        }

        if (Model.read_data_phi)
        {
            phi = Base::get_lbm_val(IJK, IPHI);
        }
        else if (Model.use_sharp_init)
        {
            phi = (Model.sign * xx) < 0 ? 0 : 1;
        }
        else
        {
            phi = Model.phi0(xx);
        }

        // get useful params
        real_t initClA = Model.initClA;
        real_t initCsA = Model.initCsA;
        real_t initClB = Model.initClB;
        real_t initCsB = Model.initCsB;
        if (Model.initType == PHASE_FIELD_RANDOM_TERNARY_GLASS or Model.initType == PHASE_FIELD_RANDOM_TERNARY_GLASS_SPHERE)
        {
            initCsB *= (1 + 0.2 * cos(6.3 * Model.freq * IJK[IX] / Base::params.nx) * cos(6.3 * Model.freq * IJK[IY] / Base::params.ny));
        }

        if (Model.initType == PHASE_FIELD_INIT_TERNARY_PERCO)
        {
            if (abs(phi - 2.0) <= 0.1)
            {

                initCsA = Model.initCsBA;
                initCsB = Model.initCsBB;

                phi = 0.0;
            }
            else if (abs(phi - 0.0) <= 0.1)
            {
                initCsA = Model.initCsAA;
                initCsB = Model.initCsAB;
            }
        }

        real_t h = Model.h(phi);
        real_t CA = (1.0 - h) * initCsA + h * initClA;
        real_t CB = (1.0 - h) * initCsB + h * initClB;

        if (Model.initType == PHASE_FIELD_INIT_VERTICAL_ERFC_1INT)
        {
            real_t t = Model.init_time;
            initClA = Model.A1A + Model.B1A * erfc(x / (2 * SQRT(t * Model.DA1)));
            initCsA = Model.A0A + Model.B0A * erfc(-x / (2 * SQRT(t * Model.DA0)));
            initClB = Model.A1B + Model.B1B * erfc(x / (2 * SQRT(t * Model.DB1)));
            initCsB = Model.A0B + Model.B0B * erfc(-x / (2 * SQRT(t * Model.DB0)));
            CA = (1.0 - h) * initCsA + h * initClA;
            CB = (1.0 - h) * initCsB + h * initClB;
            //~ CA = (Model.sign * xx) < 0 ? initCsA : initClA;
            //~ CB = (Model.sign * xx) < 0 ? initCsB : initClB;
        }

        // compute mu
        const real_t muA = Model.compute_muA(type, phi, CA);
        const real_t muB = Model.compute_muB(type, phi, CB);

        real_t init_dphidt = 0.0;
        if (Model.initType == PHASE_FIELD_INIT_VERTICAL_ERFC_1INT)
        {
            init_dphidt = 4 * phi * (1 - phi) * Model.xi / 2 / SQRT(Model.init_time);
        }

        // set macro fields values
        Base::set_lbm_val(IJK, IPHI, phi);
        Base::set_lbm_val(IJK, IC, CA);
        Base::set_lbm_val(IJK, ICB, CB);
        Base::set_lbm_val(IJK, IMU, muA);
        Base::set_lbm_val(IJK, IMUB, muB);

        // init grad phi
        Base::set_lbm_val(IJK, IDPHIDX, 0.0);
        Base::set_lbm_val(IJK, IDPHIDY, 0.0);

        // init time derivatives
        Base::set_lbm_val(IJK, IDPHIDT, init_dphidt);
        Base::set_lbm_val(IJK, IDMUDT, 0.0);

    } // end init macro

    KOKKOS_INLINE_FUNCTION
    void update_macro(const IVect<dim>& IJK) const
    {
        // get useful params
        const real_t dt = Base::params.dt;

        // store old values of macro fields
        LBMState lbmStatePrev;
        Base::setupLBMState(IJK, lbmStatePrev);

        real_t moment_phi = 0.0;
        real_t moment_C = 0.0;
        real_t moment_CB = 0.0;
        for (int ipop = 0; ipop < npop; ++ipop)
        {
            moment_phi += Base::get_f_val(tagPHI, IJK, ipop);
            moment_C += Base::get_f_val(tagC, IJK, ipop);
            moment_CB += Base::get_f_val(tagCB, IJK, ipop);
        }

        // get source terms and compute new macro fields
        const real_t source_phi = Model.S0(type, tagPHI, lbmStatePrev);

        real_t phi = FMIN(FMAX(moment_phi + 0.5 * dt * source_phi, 0.0), 1.0);
        real_t C = FMAX(moment_C, moment_C);
        real_t CB = FMAX(moment_CB, moment_CB);

        // compute new chemical potential
        const real_t mu = Model.compute_muA(type, phi, C);
        const real_t muB = Model.compute_muB(type, phi, CB);

        // update macro fields
        Base::set_lbm_val(IJK, IPHI, phi);
        Base::set_lbm_val(IJK, IC, C);
        Base::set_lbm_val(IJK, IMU, mu);
        Base::set_lbm_val(IJK, ICB, CB);
        Base::set_lbm_val(IJK, IMUB, muB);

        // update time derivatives if needed
        Base::set_lbm_val(IJK, IDPHIDT, (phi - lbmStatePrev[IPHI]) / dt);
        Base::set_lbm_val(IJK, IDMUDT, (mu - lbmStatePrev[IMU]) / dt);
    } // end update macro

    KOKKOS_INLINE_FUNCTION
    void update_macro_grad(const IVect<dim>& IJK) const
    {
        RVect<dim> gradPhi;
        Base::compute_gradient(gradPhi, IJK, IPHI, BOUNDARY_EQUATION_1);
        Base::set_lbm_val(IJK, IDPHIDX, gradPhi[IX]);
        Base::set_lbm_val(IJK, IDPHIDY, gradPhi[IY]);
        if (dim == 3)
        {
            Base::set_lbm_val(IJK, IDPHIDZ, gradPhi[IZ]);
        }

        RVect<dim> gradCA;
        Base::compute_gradient(gradCA, IJK, IC, BOUNDARY_EQUATION_2);
        Base::set_lbm_val(IJK, IDCADX, gradCA[IX]);
        Base::set_lbm_val(IJK, IDCADY, gradCA[IY]);
        if (dim == 3)
        {
            Base::set_lbm_val(IJK, IDCADZ, gradCA[IZ]);
        }

        RVect<dim> gradCB;
        Base::compute_gradient(gradCB, IJK, ICB, BOUNDARY_EQUATION_3);
        Base::set_lbm_val(IJK, IDCBDX, gradCB[IX]);
        Base::set_lbm_val(IJK, IDCBDY, gradCB[IY]);
        if (dim == 3)
        {
            Base::set_lbm_val(IJK, IDCBDZ, gradCB[IZ]);
        }

        // update additionnal outputs
        LBMState lbmStateNew;
        Base::setupLBMState(IJK, lbmStateNew);
        Base::set_lbm_val(IJK, IDELTAW, Model.compute_dw(type, lbmStateNew));

        Base::set_lbm_val(IJK, IGRANDPOTENTIAL, Model.compute_gp(type, lbmStateNew));

        Base::set_lbm_val(IJK, I_FREE_ENERGY, Model.compute_free_energy(type, lbmStateNew));

        const real_t phi = lbmStateNew[IPHI];
        RVect<dim> term;
        const real_t norm = Model.m_norm(term, lbmStateNew[IDPHIDX], lbmStateNew[IDPHIDY], lbmStateNew[IDPHIDZ]);
        const real_t ie = Model.W * Model.W * norm * norm / 2 + Model.g(phi);
        Base::set_lbm_val(IJK, IINTERFACIALENERGY, ie);
    }

    real_t nltotA;
    real_t nltotB;
    real_t vl;

    // tools to apply infinite diff on each connected component

    uset CCid_vv;
    umap map_CCid_to_phi;

    umap map_CCid_to_phiClA;
    umap map_CCid_to_phiClB;

    AtomicAdd op;

    struct FunctorCustomUpdateC
    {
        const LBMScheme<dim, npop, modelType> scheme;

        KOKKOS_INLINE_FUNCTION
        FunctorCustomUpdateC(const LBMScheme<dim, npop, modelType>& scheme)
          : scheme(scheme){};

        KOKKOS_INLINE_FUNCTION
        void operator()(const IVect<dim>& IJK) const
        {
            real_t phi = scheme.get_lbm_val(IJK, IPHI);
            real_t muA = scheme.get_lbm_val(IJK, IMU);
            real_t muB = scheme.get_lbm_val(IJK, IMUB);

            real_t CdiffA = phi * scheme.nltotA / scheme.vl + (1 - phi) * (scheme.Model.compute_csA(scheme.type, muA));
            real_t CdiffB = phi * scheme.nltotB / scheme.vl + (1 - phi) * (scheme.Model.compute_csB(scheme.type, muB));
            real_t mudiffA = scheme.Model.compute_muA(scheme.type, phi, CdiffA);
            real_t mudiffB = scheme.Model.compute_muB(scheme.type, phi, CdiffB);
            scheme.set_lbm_val(IJK, IC, CdiffA);
            scheme.set_lbm_val(IJK, ICB, CdiffB);
            scheme.set_lbm_val(IJK, IMU, mudiffA);
            scheme.set_lbm_val(IJK, IMUB, mudiffB);
        };
    };

    ////////////////////////////////////////////////////
    // functions for determination of connected components, using label diffusion method (MPAR)

    struct FunctorCustomInitCClabels
    {
        const LBMScheme<dim, npop, modelType> scheme;

        KOKKOS_INLINE_FUNCTION
        FunctorCustomInitCClabels(const LBMScheme<dim, npop, modelType>& scheme)
          : scheme(scheme){};

        KOKKOS_INLINE_FUNCTION
        void operator()(const IVect<dim>& IJK) const
        {

            //~ bool isNotGW= scheme.isNotGhostCell(IJK);
            bool isInBounds = scheme.is_in_bounds(scheme.tagPHI, IJK);
            real_t i = scheme.IJKToCCindex(IJK);
            real_t phi = scheme.get_lbm_val(IJK, IPHI);
            //~ i = (isNotGW)*(phi>scheme.Model.CC_phi_threshold)*i;
            i = (isInBounds) * (phi > scheme.Model.CC_phi_threshold) * i;
            scheme.set_cc_label(IJK, i);
        };
    };
    struct FunctorCustom_Copy_CClabels_for_output
    {
        const LBMScheme<dim, npop, modelType> scheme;

        KOKKOS_INLINE_FUNCTION
        FunctorCustom_Copy_CClabels_for_output(const LBMScheme<dim, npop, modelType>& scheme)
          : scheme(scheme){};

        KOKKOS_INLINE_FUNCTION
        void operator()(const IVect<dim>& IJK) const
        {
            real_t cclabel = scheme.get_cc_label(IJK);
            scheme.set_lbm_val(IJK, ICC, cclabel);
        };
    };

    struct FunctorCustomComputeIntegrals
    {
        const LBMScheme<dim, npop, modelType> scheme;

        KOKKOS_INLINE_FUNCTION
        FunctorCustomComputeIntegrals(const LBMScheme<dim, npop, modelType>& scheme)
          : scheme(scheme){};

        KOKKOS_INLINE_FUNCTION
        void operator()(const IVect<dim>& IJK) const
        {

            bool isNotGW = scheme.isNotGhostCell(IJK);
            if (isNotGW)
            {
                real_t dx = scheme.params.dx;
                real_t dv = dim == 2 ? dx * dx : dx * dx * dx;

                real_t phi = scheme.get_lbm_val(IJK, IPHI);
                real_t muA = scheme.get_lbm_val(IJK, IMU);
                real_t muB = scheme.get_lbm_val(IJK, IMUB);
                real_t cc = scheme.get_cc_label(IJK);
                int cci = int(cc);

                scheme.map_CCid_to_phi.insert(cci, (dv * phi), scheme.op);
                scheme.map_CCid_to_phiClA.insert(cci, (dv * phi * scheme.Model.compute_clA(scheme.type, muA)), scheme.op);
                scheme.map_CCid_to_phiClB.insert(cci, (dv * phi * scheme.Model.compute_clB(scheme.type, muB)), scheme.op);
            }

            if (scheme.Model.apply_virtual_volume)
            {
                IVect<dim> IJKvv;
                scheme.getIJKvv(IJKvv);
                bool is_vv=true;
                for (int i=0; i<dim;i++)
                {
                    is_vv = (is_vv and IJK[i]==IJKvv[i]);
                }
                if (is_vv){
                    scheme.CCid_vv.insert(scheme.get_cc_label(IJKvv));
                }
            }
        };
    };

    // apply diff
    //~ KOKKOS_INLINE_FUNCTION

    struct FunctorCustomApplyCCdiff
    {
        const LBMScheme<dim, npop, modelType> scheme;

        KOKKOS_INLINE_FUNCTION
        FunctorCustomApplyCCdiff(const LBMScheme<dim, npop, modelType>& scheme)
          : scheme(scheme){};

        KOKKOS_INLINE_FUNCTION
        void operator()(const IVect<dim>& IJK) const
        {

            real_t phi = scheme.get_lbm_val(IJK, IPHI);
            real_t muA = scheme.get_lbm_val(IJK, IMU);
            real_t muB = scheme.get_lbm_val(IJK, IMUB);
            real_t cc = scheme.get_cc_label(IJK);
            int cci = int(cc);

            if (cc > 0.5)
            {
                //~ printf("MPI %d, pos %d %d, cc %d\n", scheme.params.myRank, IJK[IX], IJK[IY], cci);

                uint32_t id1 = scheme.map_CCid_to_phi.find(cci);
                uint32_t idA = scheme.map_CCid_to_phiClA.find(cci);
                uint32_t idB = scheme.map_CCid_to_phiClB.find(cci);

                // throw not usable on cuda device
                // if (not(scheme.map_CCid_to_phi.valid_at(id1)))
                // {
                // printf("ERROR: unvalid integral phi for cci %d, at %d,%d\n", cci, IJK[IX], IJK[IY]);
                // std::cout<<std::flush;
                // throw std::runtime_error("unvalid integral phi");
                // }
                // if (not(scheme.map_CCid_to_phiClA.valid_at(idA)))
                // {
                // printf("ERROR: unvalid integral clA for cci %d, at %d,%d\n", cci, IJK[IX], IJK[IY]);
                // std::cout<<std::flush;
                // throw std::runtime_error("unvalid integral clA");
                // }
                // if (not(scheme.map_CCid_to_phiClB.valid_at(idB)))
                // {
                // printf("ERROR: unvalid integral clB for cci %d, at %d,%d\n", cci, IJK[IX], IJK[IY]);
                // std::cout<<std::flush;
                // throw std::runtime_error("unvalid integral clB");
                // }

                if (scheme.map_CCid_to_phi.valid_at(id1) and scheme.map_CCid_to_phiClA.valid_at(idA) and scheme.map_CCid_to_phiClB.valid_at(idB))
                {
                    real_t Vcc = scheme.map_CCid_to_phi.value_at(id1);
                    real_t NAcc = scheme.map_CCid_to_phiClA.value_at(idA);
                    real_t NBcc = scheme.map_CCid_to_phiClB.value_at(idB);
                    
                    // if (scheme.Model.apply_virtual_volume)
                    // {
                        // IVect<dim> IJKvv;
                        // scheme.setIJKvv(IJKvv);
                        // bool is_vv=true;
                        // for (int i=0; i<dim;i++)
                        // {
                            // is_vv = (is_vv and IJK[i]==IJKvv[i]);
                        // }
                        // if (is_vv){
                            // Vcc+=scheme.Model.virtual_volume;
                        // }
                    // }

                    real_t CdiffA = phi * NAcc / Vcc + (1.0 - phi) * (scheme.Model.compute_csA(scheme.type, muA));
                    real_t CdiffB = phi * NBcc / Vcc + (1.0 - phi) * (scheme.Model.compute_csB(scheme.type, muB));
                    real_t mudiffA = scheme.Model.compute_muA(scheme.type, phi, CdiffA);
                    real_t mudiffB = scheme.Model.compute_muB(scheme.type, phi, CdiffB);
                    scheme.set_lbm_val(IJK, IC, CdiffA);
                    scheme.set_lbm_val(IJK, ICB, CdiffB);
                    scheme.set_lbm_val(IJK, IMU, mudiffA);
                    scheme.set_lbm_val(IJK, IMUB, mudiffB);
                }

                else
                {
                }
            }
        };
    };

    struct FunctorSumPhi
    {
        const LBMScheme<dim, npop, modelType> scheme;

        KOKKOS_INLINE_FUNCTION
        FunctorSumPhi(const LBMScheme<dim, npop, modelType>& scheme)
          : scheme(scheme){};

        KOKKOS_INLINE_FUNCTION
        real_t operator()(const IVect<dim>& IJK) const
        {
            real_t dx = scheme.params.dx;
            real_t dv = dim == 2 ? dx * dx : dx * dx * dx;
            real_t phi = scheme.get_lbm_val(IJK, IPHI);

            return (dv * phi);
        };
    }; // end functor phi

    struct FunctorSumphiClA
    {
        const LBMScheme<dim, npop, modelType> scheme;

        KOKKOS_INLINE_FUNCTION
        FunctorSumphiClA(const LBMScheme<dim, npop, modelType>& scheme)
          : scheme(scheme){};

        KOKKOS_INLINE_FUNCTION
        real_t operator()(const IVect<dim>& IJK) const
        {
            real_t dx = scheme.params.dx;
            real_t dv = dim == 2 ? dx * dx : dx * dx * dx;
            real_t phi = scheme.get_lbm_val(IJK, IPHI);
            real_t mu = scheme.get_lbm_val(IJK, IMU);

            return (dv * phi * scheme.Model.compute_clA(scheme.type, mu));
        };

    }; // end functor phiClA

    struct FunctorSumphiClB
    {
        const LBMScheme<dim, npop, modelType> scheme;

        KOKKOS_INLINE_FUNCTION
        FunctorSumphiClB(const LBMScheme<dim, npop, modelType>& scheme)
          : scheme(scheme){};

        KOKKOS_INLINE_FUNCTION
        real_t operator()(const IVect<dim>& IJK) const
        {
            real_t dx = scheme.params.dx;
            real_t dv = dim == 2 ? dx * dx : dx * dx * dx;
            real_t phi = scheme.get_lbm_val(IJK, IPHI);
            real_t mu = scheme.get_lbm_val(IJK, IMUB);

            return (dv * phi * scheme.Model.compute_clB(scheme.type, mu));
        };

    }; // end functor phiClB

    struct FunctorSumCA
    {
        const LBMScheme<dim, npop, modelType> scheme;

        KOKKOS_INLINE_FUNCTION
        FunctorSumCA(const LBMScheme<dim, npop, modelType>& scheme)
          : scheme(scheme){};

        KOKKOS_INLINE_FUNCTION
        real_t operator()(const IVect<dim>& IJK) const
        {
            real_t dx = scheme.params.dx;
            real_t dv = dim == 2 ? dx * dx : dx * dx * dx;
            real_t C = scheme.get_lbm_val(IJK, IC);

            return (dv * C);
        };

    }; // end functor CA

    struct FunctorSumCB
    {
        const LBMScheme<dim, npop, modelType> scheme;

        KOKKOS_INLINE_FUNCTION
        FunctorSumCB(const LBMScheme<dim, npop, modelType>& scheme)
          : scheme(scheme){};

        KOKKOS_INLINE_FUNCTION
        real_t operator()(const IVect<dim>& IJK) const
        {
            real_t dx = scheme.params.dx;
            real_t dv = dim == 2 ? dx * dx : dx * dx * dx;
            real_t C = scheme.get_lbm_val(IJK, ICB);

            return (dv * C);
        };

    }; // end functor CB
    KOKKOS_INLINE_FUNCTION
    void getIJKvv(IVect2& IJKvv) const
    {
        int gw=Base::params.ghostWidth;
        IJKvv = IVect<2>({ gw + Model.virtual_volume_anchor_x, gw + Model.virtual_volume_anchor_y });
    }
    KOKKOS_INLINE_FUNCTION
    void getIJKvv(IVect3& IJKvv) const
    {
        int gw=Base::params.ghostWidth;
        IJKvv = IVect<3>({ 
            gw + Model.virtual_volume_anchor_x, 
            gw + Model.virtual_volume_anchor_y,
            gw + Model.virtual_volume_anchor_z});
    }

}; // end class LBMScheme_ADE_std

};
#endif // KERNELS_FUNCTORS_H_
