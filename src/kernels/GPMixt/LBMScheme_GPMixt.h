#ifndef LBMSCHEME_GPMIXT_H_
#define LBMSCHEME_GPMIXT_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>

#endif // __CUDA_ARCH__

#include "FieldManager.h"
#include "LBMParams.h"
#include "LBM_Base_Functor.h"

#include "Models_GP_Mixt.h"
namespace PBM_GP_MIXT {
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
    EquationTag2 tagC;

    LBMScheme()
        : LBMSchemeBase<dim, npop>()
        , w(LBMBaseFunctor<dim, npop>::w) {};

    LBMScheme(LBMParams& params, LBMArray lbm_data, model_t& Model)
        : LBMSchemeBase<dim, npop>(params, lbm_data)
        , lbm_data(lbm_data)
        , Model(Model)
        , w(LBMBaseFunctor<dim, npop>::w)
        , map_CCid_to_phi(uint32_t(1000))
        , map_CCid_to_phiCl(uint32_t(1000)) {};

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
    void setup_collider(EquationTag1 tag, const IVect<dim>& IJK, TRTCollider<dim, npop>& collider) const
    {

        const real_t dt = this->params.dt;
        const real_t dx = this->params.dx;
        LBMState lbmState;
        this->setupLBMState(IJK, lbmState);

        const real_t M0 = Model.M0(type, tag, lbmState);
        const real_t M2 = Model.M2(type, tag, lbmState);
        const real_t S0 = Model.S0(type, tag, lbmState);
        const RVect<dim> S1 = Model.S1<dim>(type, tag, lbmState);

        // compute collision rate
        collider.tauA = Model.tau(type, tag, lbmState);
        collider.tauS = 0.5 + this->params.lambdaTRT1 / (collider.tauA - 0.5);
        real_t staudx = 1 / ((collider.tauA - 0.5) * dx / Model.e2);

        // ipop = 0
        collider.f[0] = this->get_f_val(tag, IJK, 0);
        real_t scal = this->compute_scal(0, S1);
        collider.S0[0] = dt * w[0] * (S0 + staudx * scal);
        //~ collider.S1[0] = dt * w[0] * staudx * scal;
        //~ collider.feq[0] = M0 - (1-w[0])*M2 - 0.5*(collider.S0[0]);
        collider.feq[0] = M0 - (1 - w[0]) * M2 - 0.5 * (collider.S0[0]);

        // ipop > 0
        for (int ipop = 1; ipop < npop; ++ipop) {
            collider.f[ipop] = this->get_f_val(tag, IJK, ipop);
            scal = this->compute_scal(ipop, S1);
            collider.S0[ipop] = dt * w[ipop] * (S0 + staudx * scal);
            //~ collider.S1[ipop] = dt * w[ipop] * staudx * scal;
            collider.feq[ipop] = w[ipop] * M2 - 0.5 * (collider.S0[ipop]);
        }
    } // end of setup_collider for phase field equation

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag2 tag, const IVect<dim>& IJK, TRTCollider<dim, npop>& collider) const
    {

        const real_t dt = this->params.dt;
        const real_t dx = this->params.dx;

        LBMState lbmState;
        this->setupLBMState(IJK, lbmState);

        const real_t M0 = Model.M0(type, tag, lbmState);
        const real_t M2 = Model.M2(type, tag, lbmState);
        const real_t S0 = Model.S0(type, tag, lbmState);
        const RVect<dim> S1 = Model.S1<dim>(type, tag, lbmState);

        // compute collision rate
        collider.tauA = Model.tau(type, tag, lbmState);
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
            real_t boundary_value = (Model.compute_mu(type, 1.0, boundary_value_C) - Model.scale_chempot * Model.mueq) * Model.D1;
            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_antibounceback(tagC, faceId, IJK, ipop, boundary_value);
        }

        else if (this->params.boundary_types[BOUNDARY_EQUATION_2][faceId] == BC_ZERO_FLUX) {
            for (int ipop = 0; ipop < npop; ++ipop)
                this->compute_boundary_bounceback(tagC, faceId, IJK, ipop, 0.0);
        }
    };

    KOKKOS_INLINE_FUNCTION
    void init_macro(const IVect<dim>& IJK, const RANDOM_POOL::generator_type& rand_gen) const
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
        else if (Model.use_sharp_init) {
            phi = (Model.sign * xx) < 0 ? 0 : 1;
        } else
            phi = Model.phi0(xx);

        // compute supersaturation
        real_t h = Model.h(phi);
        const real_t C = (1.0 - h) * initCs + h * initCl;
        // compute mu
        const real_t mu = Model.compute_mu(type, phi, C);

        // set macro fields values
        if (not(Model.initType == PHASE_FIELD_INIT_DATA))
            this->set_lbm_val(IJK, IPHI, phi);
        this->set_lbm_val(IJK, IC, C);
        this->set_lbm_val(IJK, IMU, mu);

        // init grad phi
        this->set_lbm_val(IJK, IDPHIDX, 0.0);
        this->set_lbm_val(IJK, IDPHIDY, 0.0);

        // init time derivatives
        Base::set_lbm_val(IJK, IDPHIDT, phi * (1 - phi) * Model.init_dphidt);
        Base::set_lbm_val(IJK, IDMUDT, 0.0);

    } // end init macro

    KOKKOS_INLINE_FUNCTION
    void update_macro(const IVect<dim>& IJK) const
    {
        // get useful params
        const real_t dt = this->params.dt;

        // compute moments of distribution equations
        real_t moment_phi = 0.0;
        real_t moment_C = 0.0;
        for (int ipop = 0; ipop < npop; ++ipop) {
            moment_phi += this->get_f_val(tagPHI, IJK, ipop);
            moment_C += this->get_f_val(tagC, IJK, ipop);
        }

        // store old values of macro fields
        LBMState lbmStatePrev;
        Base::setupLBMState(IJK, lbmStatePrev);

        // get source terms and compute new macro fields
        const real_t source_phi = Model.S0(type, tagPHI, lbmStatePrev);
        real_t phi = moment_phi + 0.5 * dt * source_phi;
        real_t C = moment_C;

        // compute new chemical potential
        const real_t mu = Model.compute_mu(type, phi, C);

        // update macro fields
        this->set_lbm_val(IJK, IPHI, phi);
        this->set_lbm_val(IJK, IC, C);
        this->set_lbm_val(IJK, IMU, mu);
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
        if (dim == 3) {
            this->set_lbm_val(IJK, IDPHIDZ, gradPhi[IZ]);
        }

        LBMState lbmStateNew;
        Base::setupLBMState(IJK, lbmStateNew);
        Base::set_lbm_val(IJK, IGRANDPOTENTIAL, Model.compute_gp_energy(type, lbmStateNew));
        Base::set_lbm_val(IJK, IHELMOLTZENERGY, Model.compute_helmoltz_energy(type, lbmStateNew));
    }
    
    

     struct FunctorSumPhi
    {
        const LBMScheme<dim,npop, modelType> scheme;
        FunctorSumPhi(const LBMScheme<dim,npop, modelType>& scheme):
            scheme(scheme) {};

        KOKKOS_INLINE_FUNCTION
        real_t operator ()(const IVect<dim>& IJK) const
        {
            real_t dx=scheme.params.dx;
            real_t dv=dim==2 ? dx*dx : dx*dx*dx;
            real_t phi = scheme.get_lbm_val(IJK,IPHI);

            return (dv*phi);
        };
    };// end functor phi
    
     struct FunctorSumPhiCl
    {
        const LBMScheme<dim,npop, modelType> scheme;
        FunctorSumPhiCl(const LBMScheme<dim,npop, modelType>& scheme):
            scheme(scheme) {};

        KOKKOS_INLINE_FUNCTION
        real_t operator ()(const IVect<dim>& IJK) const
        {
            real_t dx=scheme.params.dx;
            real_t dv=dim==2 ? dx*dx : dx*dx*dx;
            real_t phi = scheme.get_lbm_val(IJK,IPHI);
            real_t mu = scheme.get_lbm_val(IJK,IMU);
            
            return (dv*phi*scheme.Model.compute_cl(scheme.type,mu));
        };
    };// end functor phi

    
         struct FunctorSumC
    {
        const LBMScheme<dim,npop, modelType> scheme;
        FunctorSumC(const LBMScheme<dim,npop, modelType>& scheme):
            scheme(scheme) {};

        KOKKOS_INLINE_FUNCTION
        real_t operator ()(const IVect<dim>& IJK) const
        {
            real_t dx=scheme.params.dx;
            real_t dv=dim==2 ? dx*dx : dx*dx*dx;
            real_t C = scheme.get_lbm_val(IJK,IC);

            return (dv*C);
        };
    };// end functor phi

    

    
    real_t nltot;
    real_t vl;
    
    struct FunctorCustomUpdateC
    {
        const LBMScheme<dim,npop, modelType> scheme;
        FunctorCustomUpdateC(const LBMScheme<dim,npop, modelType>& scheme):
            scheme(scheme) {};

        KOKKOS_INLINE_FUNCTION
        void operator ()(const IVect<dim>& IJK) const
        {
            real_t phi = scheme.get_lbm_val(IJK,IPHI);
            real_t mu = scheme.get_lbm_val(IJK,IMU);



            real_t Cdiff = phi*scheme.nltot/scheme.vl + (1-phi)*(scheme.Model.compute_cs(scheme.type,mu));
  
            real_t mudiff = scheme.Model.compute_mu(scheme.type,phi,Cdiff);
 
            scheme.set_lbm_val(IJK, IC, Cdiff);
            scheme.set_lbm_val(IJK, IMU, mudiff);

        };

    };
    
    ////////////////////////////////////////////////////
    // functions for determination of connected components, using label diffusion method (MPAR)
    

    
    // init labels for connected component determination    
    struct FunctorCustomInitCClabels
    {
        const LBMScheme<dim,npop, modelType> scheme;
        FunctorCustomInitCClabels(const LBMScheme<dim,npop, modelType>& scheme):
            scheme(scheme) {};

        KOKKOS_INLINE_FUNCTION
        void operator ()(const IVect<dim>& IJK) const
        {

            //~ bool isNotGW= scheme.isNotGhostCell(IJK);
            bool isInBounds= scheme.is_in_bounds(scheme.tagPHI,IJK);
            real_t i = scheme.IJKToCCindex(IJK);
            real_t phi = scheme.get_lbm_val(IJK,IPHI);
            //~ i = (isNotGW)*(phi>scheme.Model.CC_phi_threshold)*i;
            i = (isInBounds)*(phi>scheme.Model.CC_phi_threshold)*i;
            scheme.set_cc_label(IJK, i);
        };

    };
    
    struct FunctorCustom_Copy_CClabels_for_output
    {
        const LBMScheme<dim,npop, modelType> scheme;
        
        KOKKOS_INLINE_FUNCTION
        FunctorCustom_Copy_CClabels_for_output(const LBMScheme<dim,npop, modelType>& scheme):
            scheme(scheme) {};

        KOKKOS_INLINE_FUNCTION
        void operator ()(const IVect<dim>& IJK) const
        {
            real_t cclabel=scheme.get_cc_label(IJK);
            scheme.set_lbm_val(IJK, ICC, cclabel);
        };

    };
   
    
    
    // tools to apply infinite diff on each connected component
    umap map_CCid_to_phi;
    umap map_CCid_to_phiCl;
    using value_view_type = Kokkos::View<real_t*, Device>;
    using map_op_type     = Kokkos::UnorderedMapInsertOpTypes<value_view_type, uint32_t>;
    using atomic_add_type = typename map_op_type::AtomicAdd;
    atomic_add_type op;



    
    
    struct FunctorCustomComputeIntegrals
    {
        const LBMScheme<dim,npop, modelType> scheme;
        FunctorCustomComputeIntegrals(const LBMScheme<dim,npop, modelType>& scheme):
            scheme(scheme) {};

        KOKKOS_INLINE_FUNCTION
        void operator ()(const IVect<dim>& IJK) const
        {
            


            bool isNotGW= scheme.isNotGhostCell(IJK);
            if (isNotGW) {
                real_t phi = scheme.get_lbm_val(IJK,IPHI);
                real_t mu = scheme.get_lbm_val(IJK,IMU);
                real_t cc = scheme.get_cc_label(IJK);
                int cci=int(cc);
    
    
                scheme.map_CCid_to_phi.insert(cci, (phi), scheme.op);
                scheme.map_CCid_to_phiCl.insert(cci, (phi*scheme.Model.compute_cl(scheme.type,mu)), scheme.op);

            }
        };

    };

    // apply diff
    //~ KOKKOS_INLINE_FUNCTION


    struct FunctorCustomApplyCCdiff
    {
        const LBMScheme<dim,npop, modelType> scheme;
        FunctorCustomApplyCCdiff(const LBMScheme<dim,npop, modelType>& scheme):
            scheme(scheme) {};

        KOKKOS_INLINE_FUNCTION
        void operator ()(const IVect<dim>& IJK) const
        {
            
        real_t phi = scheme.get_lbm_val(IJK,IPHI);
        real_t mu = scheme.get_lbm_val(IJK,IMU);
        real_t cc = scheme.get_cc_label(IJK);
        
        if (cc>0.5)
        {
            uint32_t id1=scheme.map_CCid_to_phi.find(int(cc));        
            uint32_t id2=scheme.map_CCid_to_phiCl.find(int(cc));
        
         
            if (scheme.map_CCid_to_phi.valid_at(id1) and scheme.map_CCid_to_phiCl.valid_at(id2)) 
            {
                
                real_t Vcc=scheme.map_CCid_to_phi.value_at(id1);
                real_t Ncc=scheme.map_CCid_to_phiCl.value_at(id2);
                
                real_t Cdiff = phi*Ncc/Vcc + (1.0-phi)*(scheme.Model.compute_cs(scheme.type,mu));
                real_t mudiff = scheme.Model.compute_mu(scheme.type,phi,Cdiff);
                scheme.set_lbm_val(IJK, IC, Cdiff);
                scheme.set_lbm_val(IJK, IMU, mudiff);
            
            }
        }

           
        };

    };


}; // end class LBMScheme_ADE_std

} //end namespace
#endif // KERNELS_FUNCTORS_H_
