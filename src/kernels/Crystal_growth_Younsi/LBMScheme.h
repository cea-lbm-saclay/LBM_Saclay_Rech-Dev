#ifndef LBMSCHEME_GPMIXT_H_
#define LBMSCHEME_GPMIXT_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>

#endif // __CUDA_ARCH__

#include "FieldManager.h"
#include "LBMParams.h"
#include "LBM_Base_Functor.h"

#include "Models.h"
namespace PBM_DIRECTIONAL_SOLIDIFICATION {
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

    using model_t = ModelParams;

    static constexpr real_t NORMAL_EPSILON = 1.0e-16;

    LBMArray lbm_data;
    model_t Model;
    modelType type;
    LBM_Weights w;
    EquationTag1 tagPHI;
    EquationTag2 tagSUPERSAT;

    LBMScheme()
        : LBMSchemeBase<dim, npop>()
        , w(LBMBaseFunctor<dim, npop>::w) {};

    LBMScheme(LBMParams& params, LBMArray lbm_data, model_t& Model)
        : LBMSchemeBase<dim, npop>(params, lbm_data)
        , lbm_data(lbm_data)
        , Model(Model)
        , w(LBMBaseFunctor<dim, npop>::w) {};

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag1 tag, const IVect<dim>& IJK, BGKCollider<dim, npop>& collider) const
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
    void setup_collider(EquationTag1 tag, const IVect<dim>& IJK, BGKColliderTimeFactor<dim, npop>& collider) const
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
        collider.factor=Model.TimeFactor(type, tag, lbmState);
        real_t staudx = 1 / ((collider.tau - 0.5) * dx / e2);

        // ipop = 0
        collider.f[0] = Base::get_f_val(tag, IJK, 0);
        real_t scal = Base::compute_scal(0, S1);
        collider.S0[0] = dt * w[0] * (S0 + staudx * scal);
        collider.feq[0] = M0 - (1 - w[0]) * M2 - 0*0.5 * collider.S0[0]/ collider.factor;
        
		IVect<dim> IJKs;
       
		bool isInBound = Base::stream_alldir(IJK, IJKs, 0);
		if (isInBound){
			collider.f_nonlocal[0]=Base::get_f_val(tag,IJKs,0);
		} else {
			collider.f_nonlocal[0]=0;
		}
		
		

        // ipop > 0
        for (int ipop = 1; ipop < npop; ++ipop) {
            collider.f[ipop] = this->get_f_val(tag, IJK, ipop);
            scal = Base::compute_scal(ipop, S1);
            collider.S0[ipop] = dt * w[ipop] * (S0 + staudx * scal);
            collider.feq[ipop] = w[ipop] * M2 - 0*0.5 * collider.S0[0]/ collider.factor;
            
			bool isInBound = Base::stream_alldir(IJK, IJKs, ipop);
			if (isInBound){
				collider.f_nonlocal[ipop]=Base::get_f_val(tag,IJKs,ipop);
			} else {
				collider.f_nonlocal[ipop]=0;
			}
        }
    } // end of setup_collider for phase field equation

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag2 tag, const IVect<dim>& IJK, BGKCollider<dim, npop>& collider) const
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
    void make_boundary(const IVect<dim>& IJK, int faceId) const
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
            real_t boundary_value_C = this->params.boundary_values[BOUNDARY_CONCENTRATION][faceId];
            real_t boundary_value = boundary_value_C;
            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_antibounceback(tagSUPERSAT, faceId, IJK, ipop, boundary_value);
        }

        else if (this->params.boundary_types[BOUNDARY_EQUATION_2][faceId] == BC_ZERO_FLUX) {
            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_bounceback(tagSUPERSAT, faceId, IJK, ipop, 0.0);
        }
    };

    KOKKOS_INLINE_FUNCTION
    void init_macro(const IVect<dim>& IJK, const RANDOM_POOL::generator_type& rand_gen) const
    {

        // get local coordinates
        real_t x, y;
        this->get_coordinates(IJK, x, y);


        real_t xx = 0.0;
        real_t phi = 0.0;
        // compute phase field
        if (Model.initType == PHASE_FIELD_INIT_VERTICAL)
            xx = x - Model.x0;
        else if (Model.initType == PHASE_FIELD_INIT_HORIZONTAL)
            xx = y - Model.y0;
        else if (Model.initType == PHASE_FIELD_INIT_COSINUS) {
			real_t Lx=(Base::params.xmax-Base::params.xmin);
			real_t yI=(Model.y0 + Model.A * cos(3.1415 * (Model.decal + Model.kx/Lx * x)));
            xx = y - yI;
            
            // compute distance projected to the direction normal to the interface at same
            if (Model.use_projected_normal_distance)
            {
				real_t dyIdx= - Model.A * sin(3.1415 * (Model.decal + Model.kx/Lx * x)) * 3.1415 * Model.kx/Lx;
				xx= (y- yI) /(SQR(dyIdx)+1) * SQRT(1+SQR(dyIdx));
			}
		}
        

        if (Model.initType == PHASE_FIELD_INIT_DATA)
            phi = this->get_lbm_val(IJK, IPHI);
        else if (Model.use_sharp_init) {
            phi = (Model.sign * xx) < 0 ? 0 : 1;
        } else
            phi = Model.phi0(xx);

        // compute supersaturation
        //~ real_t h = Model.h(phi);
        const real_t U =  Model.initU;

		real_t chi=(y-Model.Vp*Model.time)/Model.lT;
        Base::set_lbm_val(IJK, ICHI, chi);

        // set macro fields values
        if (not(Model.initType == PHASE_FIELD_INIT_DATA))
            this->set_lbm_val(IJK, IPHI, phi);
        this->set_lbm_val(IJK, ISUPERSAT, U);

        // init grad phi
        this->set_lbm_val(IJK, IDPHIDX, 0.0);
        this->set_lbm_val(IJK, IDPHIDY, 0.0);

        // init time derivatives
        Base::set_lbm_val(IJK, IDPHIDT, Model.Vp/Model.W);

    } // end init macro

    KOKKOS_INLINE_FUNCTION
    void update_macro(const IVect<dim>& IJK) const
    {
        // get useful params
        const real_t dt = this->params.dt;

        // compute moments of distribution equations
        real_t moment_phi = 0.0;
        real_t moment_SUPERSAT = 0.0;
        for (int ipop = 0; ipop < npop; ++ipop) {
            moment_phi += this->get_f_val(tagPHI, IJK, ipop);
            moment_SUPERSAT += this->get_f_val(tagSUPERSAT, IJK, ipop);
        }

        // store old values of macro fields
        LBMState lbmStatePrev;
        Base::setupLBMState(IJK, lbmStatePrev);

        // get source terms and compute new macro fields
        const real_t source_phi = Model.S0(type, tagPHI, lbmStatePrev);
        const real_t source_ss = Model.S0(type, tagSUPERSAT, lbmStatePrev);
        const real_t factor = Model.TimeFactor(type, tagPHI, lbmStatePrev);
        real_t phi = (moment_phi + 0*0.5 *dt* source_phi/ factor);
        real_t supersat = moment_SUPERSAT+ 0.5 * dt * source_ss;


        // update macro fields
        this->set_lbm_val(IJK, IPHI, phi);
        this->set_lbm_val(IJK, ISUPERSAT, supersat);

        // update time derivatives if needed

        Base::set_lbm_val(IJK, IDPHIDT, (phi - lbmStatePrev[IPHI]) / dt);
        
        // get local coordinates
        real_t x, y;
        this->get_coordinates(IJK, x, y);
        
        real_t chi=(y-Model.Vp*Model.time)/Model.lT;
        Base::set_lbm_val(IJK, ICHI, chi);

    } // end update macro

    KOKKOS_INLINE_FUNCTION
    void update_macro_grad(const IVect<dim>& IJK) const
    {
        RVect<dim> gradPhi;
        Base::compute_gradient(gradPhi, IJK, IPHI, BOUNDARY_EQUATION_1);
        Base::set_lbm_val(IJK, IDPHIDX, gradPhi[IX]);
        Base::set_lbm_val(IJK, IDPHIDY, gradPhi[IY]);
        if (dim == 3) {
            this->set_lbm_val(IJK, IDPHIDZ, gradPhi[IZ]);
        }
		real_t laplaPhi = Base::compute_laplacian(IJK, IPHI, BOUNDARY_EQUATION_1);
        Base::set_lbm_val(IJK, ILAPLAPHI, laplaPhi);
        

    }

}; // end class LBMScheme_ADE_std

} //end namespace
#endif // KERNELS_FUNCTORS_H_
