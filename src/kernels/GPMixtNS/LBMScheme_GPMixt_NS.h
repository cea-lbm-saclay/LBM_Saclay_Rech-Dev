#ifndef LBMSCHEME_GPMIXT_NS_H_
#define LBMSCHEME_GPMIXT_NS_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>

#endif // __CUDA_ARCH__

#include "FieldManager.h"
#include "LBMParams.h"
#include "LBM_Base_Functor.h"

#include "Models_GP_Mixt_NS.h"
namespace PBM_GP_MIXT_NS {
template <int dim, int npop, typename modelType>
struct LBMScheme : public LBMSchemeBase<dim, npop> {
    using Base = LBMSchemeBase<dim, npop>;

    using LBMState = typename Kokkos::Array<real_t, COMPONENT_SIZE>;
    using id2index_t = Kokkos::Array<int, COMPONENT_SIZE>;
    using LBMArray = typename LBMBaseFunctor<dim, npop>::LBMArray;
    using FArray = typename LBMBaseFunctor<dim, npop>::FArray;
    using LBM_Weights = typename LBMBaseFunctor<dim, npop>::LBM_Weights;
    using LBM_speeds = typename LBMBaseFunctor<dim, npop>::LBM_speeds;
    using FState = typename Kokkos::Array<real_t, npop>;
    //~ using namespace localDataTypes;
    using Collider = BGKCollider<npop>;
    static constexpr real_t NORMAL_EPSILON = 1.0e-16;

    id2index_t fm;
    LBMArray lbm_data;
    ModelParams Model;
    modelType type;
    LBM_Weights w;
    EquationTag1 tagPHI;
    EquationTag2 tagC;
    EquationTag3 tagNS;

    LBMScheme()
        : LBMSchemeBase<dim, npop>()
        , w(LBMBaseFunctor<dim, npop>::w) {};

    LBMScheme(LBMParams params, LBMArray lbm_data, ModelParams& Model)
        : LBMSchemeBase<dim, npop>(params, lbm_data)
        , lbm_data(lbm_data)
        , Model(Model)
        , w(LBMBaseFunctor<dim, npop>::w) {};

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag1 tag, IVect<dim> IJK, Collider& collider) const
    {

        const real_t dt = Base::params.dt;
        const real_t dx = this->params.dx;
        const real_t e2 = Model.e2;

        LBMState lbmState;
        Base::setupLBMState(IJK, lbmState);

        const real_t M0 = Model.M0(type, tag, lbmState);
        const real_t M2 = Model.M2(type, tag, lbmState);
        const real_t S0 = Model.S0(type, tag, lbmState);
        const RVect<dim> S1 = Model.S1<dim>(type, tag, lbmState);

        // compute collision rate
        collider.tau = Model.tau(type, tag, lbmState);
        real_t staudx = 1 / ((collider.tau - 0.5) * dx / e2);

        // ipop = 0
        collider.f[0] = Base::get_f_val(tag, IJK, 0);
        real_t scal = Base::compute_scal(0, S1);
        collider.S0[0] = dt * w[0] * (S0 + staudx * scal);
        collider.feq[0] = M0 - (1 - w[0]) * M2 - 0.5 * collider.S0[0];

        // ipop > 0
        for (int ipop = 1; ipop < npop; ++ipop) {
            collider.f[ipop] = this->get_f_val(tag, IJK, ipop);
            scal = Base::compute_scal(ipop, S1);
            collider.S0[ipop] = dt * w[ipop] * (S0 + staudx * scal);
            collider.feq[ipop] = w[ipop] * M2 - 0.5 * (collider.S0[ipop]);
        }
    } // end of setup_collider for phase field equation

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag2 tag, IVect<dim> IJK, BGKCollider<npop>& collider) const
    {

        const real_t dt = Base::params.dt;
        const real_t dx = Base::params.dx;

        LBMState lbmState;
        Base::setupLBMState(IJK, lbmState);

        const real_t M0 = Model.M0(type, tag, lbmState);
        const real_t M2 = Model.M2(type, tag, lbmState);
        const real_t S0 = Model.S0(type, tag, lbmState);
        const RVect<dim> S1 = Model.S1<dim>(type, tag, lbmState);

        // compute collision rates
        collider.tau = Model.tau(type, tag, lbmState);
        real_t staudx = 1 / ((collider.tau - 0.5) * dx / Model.e2);

        // ipop = 0
        collider.f[0] = Base::get_f_val(tag, IJK, 0);
        real_t scal = Base::compute_scal(0, S1);
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

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag3 tag, IVect<dim> IJK, Collider& collider) const
    {

        const real_t dx = this->params.dx;
        const real_t dt = this->params.dt;
        const real_t cs2 = SQR(dx / dt) / Model.e2;
        const real_t c = dx / dt;

        LBMState lbmState;
        this->setupLBMState(IJK, lbmState);

        FState GAMMA;

        RVect2 force = Model.force_NS(type, lbmState);
        RVect2 forceP = Model.force_P(type, lbmState);
        RVect2 cmu;

        real_t scalUU = SQR(lbmState[fm[IU]]) + SQR(lbmState[fm[IV]]);
        // compute collision rate
        collider.tau = Model.tau_NS(type, lbmState);
        for (int ipop = 0; ipop < npop; ++ipop) {
            cmu[IX] = (c * this->E[ipop][IX] - lbmState[fm[IU]]);
            cmu[IY] = (c * this->E[ipop][IY] - lbmState[fm[IV]]);
            real_t scalUC = dx / dt * this->compute_scal(ipop, lbmState[fm[IU]], lbmState[fm[IV]]);
            GAMMA[ipop] = scalUC / cs2 + 0.5 * SQR(scalUC) / SQR(cs2) - 0.5 * scalUU / cs2;
            collider.f[ipop] = this->get_f_val(tag, IJK, ipop);
            collider.S0[ipop] = dt * w[ipop] * ((cmu[IX] * force[IX] + cmu[IY] * force[IY]) * (1 + GAMMA[ipop]) + (cmu[IX] * forceP[IX] + cmu[IY] * forceP[IY]) * (GAMMA[ipop]));
            collider.feq[ipop] = w[ipop] * (lbmState[fm[IP]] + lbmState[fm[ID]] * cs2 * GAMMA[ipop]) - 0.5 * (collider.S0[ipop]);
        }
    } // end of setup_collider for composition equation

    KOKKOS_INLINE_FUNCTION
    void make_boundary(IVect<dim> IJK, int faceId) const
    {

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

        // C boundary
        if (this->params.boundary_types[BOUNDARY_EQUATION_2][faceId] == BC_ANTI_BOUNCE_BACK) {
            real_t boundary_value = this->params.boundary_values[BOUNDARY_CONCENTRATION][faceId];
            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_antibounceback(tagC, faceId, IJK, ipop, boundary_value);
        }

        else if (this->params.boundary_types[BOUNDARY_EQUATION_2][faceId] == BC_ZERO_FLUX) {
            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_bounceback(tagC, faceId, IJK, ipop, 0.0);
        }

        // NS boundaries
        if (this->params.boundary_types[BOUNDARY_EQUATION_3][faceId] == BC_ANTI_BOUNCE_BACK) {
            real_t boundary_value = this->params.boundary_values[BOUNDARY_PRESSURE][faceId];
            for (int ipop = 0; ipop < npop; ++ipop) {
                this->compute_boundary_antibounceback(tagNS, faceId, IJK, ipop, boundary_value);
            }
        }

        else if (this->params.boundary_types[BOUNDARY_EQUATION_3][faceId] == BC_ZERO_FLUX) {
            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_bounceback(tagNS, faceId, IJK, ipop, 0.0);

        }

        else if (this->params.boundary_types[BOUNDARY_EQUATION_3][faceId] == BC_BOUNCEBACK) {
            const real_t dx = this->params.dx;
            const real_t dt = this->params.dt;
            const real_t cs2 = SQR(dx / dt) / Model.e2;
            real_t boundary_vx = this->params.boundary_values[BOUNDARY_VELOCITY_X][faceId];
            real_t boundary_vy = this->params.boundary_values[BOUNDARY_VELOCITY_Y][faceId];

            real_t value;
            for (int ipop = 0; ipop < npop; ++ipop) {
                value = dx / dt * this->compute_scal_Ebar(ipop, boundary_vx, boundary_vy) / cs2 * Model.rhoS;
                this->compute_boundary_bounceback(tagNS, faceId, IJK, ipop, value);
            }

        } else if (this->params.boundary_types[BOUNDARY_EQUATION_3][faceId] == BC_POISEUILLE) {
            const real_t dx = this->params.dx;
            const real_t dt = this->params.dt;
            const real_t isize = this->params.isize;
            const real_t jsize = this->params.jsize;
            const real_t cs2 = SQR(dx / dt) / Model.e2;
            real_t scaling = (faceId == FACE_YMIN or faceId == FACE_YMAX) * 4 * IJK[IX] * (isize - IJK[IX]) / SQR(isize)
                + (faceId == FACE_XMIN or faceId == FACE_XMAX) * 4 * IJK[IY] * (jsize - IJK[IY]) / SQR(jsize);
            real_t boundary_vx = scaling * this->params.boundary_values[BOUNDARY_VELOCITY_X][faceId];
            real_t boundary_vy = scaling * this->params.boundary_values[BOUNDARY_VELOCITY_Y][faceId];

            real_t value;
            for (int ipop = 0; ipop < npop; ++ipop) {
                value = dx / dt * this->compute_scal_Ebar(ipop, boundary_vx, boundary_vy) / cs2 * Model.rhoS;
                this->compute_boundary_bounceback(tagNS, faceId, IJK, ipop, value);
            }
        }
    };

    KOKKOS_INLINE_FUNCTION
    void init_macro(IVect<dim> IJK, RANDOM_POOL::generator_type rand_gen) const
    {

        // get local coordinates
        real_t x, y;
        this->get_coordinates(IJK, x, y);

        // get useful params
        const real_t initCl = Model.initCl;
        const real_t initCs = Model.initCs;
        real_t xx = 0.0;
        real_t phi = 0.0;
        // compute phase field
        if (Model.initType == PHASE_FIELD_INIT_VERTICAL)
            xx = x - Model.x0;
        else if (Model.initType == PHASE_FIELD_INIT_SPHERE)
            xx = (Model.r0 - sqrt(SQR(x - Model.x0) + SQR(y - Model.y0)));
        else if (Model.initType == PHASE_FIELD_INIT_SQUARE)
            xx = (Model.r0 - FMAX(sqrt(SQR(x - Model.x0)), sqrt(SQR(y - Model.y0))));
        else if (Model.initType == PHASE_FIELD_INIT_MIXTURE)
            xx = x - Model.x0;

        if (Model.initType == PHASE_FIELD_INIT_DATA)
            phi = this->get_lbm_val(IJK, fm[IPHI]);
        else
            phi = Model.phi0(xx);

        // compute supersaturation
        real_t h = Model.h(phi);
        const real_t C = (1.0 - h) * initCs + h * initCl;
        // compute mu
        const real_t mu = Model.compute_mu(type, phi, C);

        // set macro fields values
        if (!(Model.initType == PHASE_FIELD_INIT_DATA))
            this->set_lbm_val(IJK, fm[IPHI], phi);
        this->set_lbm_val(IJK, fm[IC], C);
        this->set_lbm_val(IJK, fm[IMU], mu);

        // init grad phi
        this->set_lbm_val(IJK, fm[IDPHIDX], 0.0);
        this->set_lbm_val(IJK, fm[IDPHIDY], 0.0);

        // init NS

        this->set_lbm_val(IJK, fm[IP], 0.0);
        this->set_lbm_val(IJK, fm[ID], Model.interp_rho(phi));
        this->set_lbm_val(IJK, fm[IU], Model.initVX);
        this->set_lbm_val(IJK, fm[IV], Model.initVY);

        // init time derivatives
        if (fm[IDPHIDT] >= 0) {
            this->set_lbm_val(IJK, fm[IDPHIDT], 0.0);
        }
        if (fm[IDMUDT] >= 0) {
            this->set_lbm_val(IJK, fm[IDMUDT], 0.0);
        }

    } // end init macro

    KOKKOS_INLINE_FUNCTION
    void update_macro(IVect<dim> IJK) const
    {
        // get useful params
        const real_t dx = this->params.dx;
        const real_t dt = this->params.dt;
        const real_t cs2 = SQR(dx / dt) / Model.e2;

        // compute moments of distribution equations
        real_t moment_phi = 0.0;
        real_t moment_C = 0.0;
        real_t moment_P = 0.0;
        real_t moment_VX = 0.0;
        real_t moment_VY = 0.0;
        for (int ipop = 0; ipop < npop; ++ipop) {
            moment_phi += this->get_f_val(tagPHI, IJK, ipop);
            moment_C += this->get_f_val(tagC, IJK, ipop);
            moment_P += this->get_f_val(tagNS, IJK, ipop);
            moment_VX += this->get_f_val(tagNS, IJK, ipop) * this->E[ipop][IX];
            moment_VY += this->get_f_val(tagNS, IJK, ipop) * this->E[ipop][IY];
        }

        // store old values of macro fields
        LBMState lbmStatePrev;
        this->setupLBMState(IJK, lbmStatePrev);

        const real_t rhoprev = lbmStatePrev[fm[ID]];

        const RVect2 force = Model.force_NS(type, lbmStatePrev);
        const RVect2 forceP = Model.force_P(type, lbmStatePrev);
        const real_t scalP = lbmStatePrev[fm[IU]] * forceP[IX] + lbmStatePrev[fm[IV]] * forceP[IY];

        const real_t source_phi = Model.S0(type, tagPHI, lbmStatePrev);
        // get source terms and compute new macro fields

        const real_t phi = moment_phi + 0.5 * dt * source_phi;
        const real_t rho = Model.interp_rho(phi);
        const real_t C = moment_C;
        const real_t mu = Model.compute_mu(type, phi, C);
        // compute NS macro vars
        const real_t P = moment_P + 0.5 * dt * scalP;
        const real_t VX = (moment_VX / cs2 + 0.5 * dt * force[IX]) / rhoprev;
        const real_t VY = (moment_VY / cs2 + 0.5 * dt * force[IY]) / rhoprev;
        // update macro fields
        this->set_lbm_val(IJK, fm[IPHI], phi);
        this->set_lbm_val(IJK, fm[ID], rho);
        this->set_lbm_val(IJK, fm[IC], C);
        this->set_lbm_val(IJK, fm[IMU], mu);
        this->set_lbm_val(IJK, fm[IP], P);
        this->set_lbm_val(IJK, fm[IU], VX);
        this->set_lbm_val(IJK, fm[IV], VY);

        // update time derivatives if needed
        if (fm[IDPHIDT] >= 0) {
            this->set_lbm_val(IJK, fm[IDPHIDT], (phi - lbmStatePrev[IPHI]) / dt);
        }
        if (fm[IDMUDT] >= 0) {
            this->set_lbm_val(IJK, fm[IDPHIDT], (mu - lbmStatePrev[IMU]) / dt);
        }
    } // end update macro

    KOKKOS_INLINE_FUNCTION
    void update_macro_grad(IVect<dim>& IJK) const
    {
        RVect<dim> gradPhi;
        this->compute_gradient(gradPhi, IJK, fm[IPHI], BOUNDARY_EQUATION_1);
        this->set_lbm_val(IJK, fm[IDPHIDX], gradPhi[IX]);
        this->set_lbm_val(IJK, fm[IDPHIDY], gradPhi[IY]);
        if (dim == 3) {
            this->set_lbm_val(IJK, fm[IDPHIDZ], gradPhi[IZ]);
        }
    }

}; // end class LBMScheme_ADE_std

} //end namespace
#endif // KERNELS_FUNCTORS_H_
