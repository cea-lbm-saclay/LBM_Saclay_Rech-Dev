/**
 * \file LBMScheme_NS_AC.h
 * \brief Scheme
 *
 */

#ifndef LBMSCHEME_NS_PF_H_
#define LBMSCHEME_NS_PF_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>

#endif // __CUDA_ARCH__

//~ #include "LBMParams.h"
//~ #include "LBM_Base_Functor.h"

#include "Models_NS_AC.h"

namespace PBM_NSAC {
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
    //~ using namespace localDataTypes;
    //~ using w =Base::w;

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
        , w(LBMBaseFunctor<dim, npop>::w) 
        , map_CCid_to_volume(KOKKOS_MAP_DEFAULT_CAPACITY)
        , map_CCid_to_vx(KOKKOS_MAP_DEFAULT_CAPACITY)
        , map_CCid_to_vy(KOKKOS_MAP_DEFAULT_CAPACITY)
        , map_CCid_to_vz(KOKKOS_MAP_DEFAULT_CAPACITY)
        , map_CCid_to_p(KOKKOS_MAP_DEFAULT_CAPACITY)
        , map_CCid_to_Fx(KOKKOS_MAP_DEFAULT_CAPACITY)
        , map_CCid_to_Fy(KOKKOS_MAP_DEFAULT_CAPACITY)
        , map_CCid_to_Fz(KOKKOS_MAP_DEFAULT_CAPACITY)
        {};

    KOKKOS_INLINE_FUNCTION
    void setup_collider(EquationTag1 tag, const IVect<dim>& IJK, BGKCollider<dim, npop>& collider) const
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
    void setup_collider(EquationTag2 tag, IVect<dim> IJK, BGKCollider<dim, npop>& collider) const
    {

        const real_t dx = Model.dx;
        const real_t dt = Model.dt;
        const real_t cs2 = SQR(dx / dt) / Model.e2;
        const real_t c = dx / dt;

        LBMState lbmState;
        Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmState);

        FState GAMMA;

        RVect2 force = Model.force_NS(type, lbmState);
        RVect2 forceP = Model.force_P(type, lbmState);
        RVect2 cmu;

        real_t scalUU = SQR(lbmState[IU]) + SQR(lbmState[IV]);
        // compute collision rate
        collider.tau = Model.tau_NS(type, lbmState);
        for (int ipop = 0; ipop < npop; ++ipop) {
            cmu[IX] = (c * E[ipop][IX] - lbmState[IU]);
            cmu[IY] = (c * E[ipop][IY] - lbmState[IV]);
            real_t scalUC = dx / dt * Base::compute_scal(ipop, lbmState[IU], lbmState[IV]);
            GAMMA[ipop] = scalUC / cs2 + 0.5 * SQR(scalUC) / SQR(cs2) - 0.5 * scalUU / cs2;
            collider.f[ipop] = Base::get_f_val(tag, IJK, ipop);
            collider.S0[ipop] = dt * w[ipop] * ((cmu[IX] * force[IX] + cmu[IY] * force[IY]) * (1 + GAMMA[ipop]) + (cmu[IX] * forceP[IX] + cmu[IY] * forceP[IY]) * (GAMMA[ipop]));
            collider.feq[ipop] = w[ipop] * (lbmState[IP] + lbmState[ID] * cs2 * GAMMA[ipop]) - 0.5 * (collider.S0[ipop]);
        }
    } // end of setup_collider for composition equation

    KOKKOS_INLINE_FUNCTION
    void make_boundary(const IVect<dim>& IJK, int faceId) const
    {
		
		int d=1;
		IVect<dim> IJKb;
		IJKb[IX]=IJK[IX]- d * (faceId==FACE_XMAX) + d * (faceId==FACE_XMIN);
		IJKb[IY]=IJK[IY]- d * (faceId==FACE_YMAX) + d * (faceId==FACE_YMIN);
		if (dim==3)
			IJKb[IZ]=IJK[IZ]- d * (faceId==FACE_ZMAX) + d * (faceId==FACE_ZMIN);

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
        else if (Base::params.boundary_types[BOUNDARY_EQUATION_1][faceId] == BC_FREEFLOW) {
			IVect<dim> IJKm2;
			int d=2;
			IJKm2[IX]=IJK[IX]- d * (faceId==FACE_XMAX) + d * (faceId==FACE_XMIN);
			IJKm2[IY]=IJK[IY]- d * (faceId==FACE_YMAX) + d * (faceId==FACE_YMIN);
			if (dim==3)
				IJKm2[IZ]=IJK[IZ]- d * (faceId==FACE_ZMAX) + d * (faceId==FACE_ZMIN);
			d=1;
			IVect<dim> IJKm1;
			IJKm1[IX]=IJK[IX]- d * (faceId==FACE_XMAX) + d * (faceId==FACE_XMIN);
			IJKm1[IY]=IJK[IY]- d * (faceId==FACE_YMAX) + d * (faceId==FACE_YMIN);
			if (dim==3)
				IJKm1[IZ]=IJK[IZ]- d * (faceId==FACE_ZMAX) + d * (faceId==FACE_ZMIN);
            real_t value;
            for (int ipop = 0; ipop < npop; ++ipop) {

                value = this->get_f_val(tagPHI, IJKm2, ipop);
                this->set_f_val(tagPHI, IJKm1, ipop, value);
            }
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
            [[maybe_unused]] const real_t cs2 = SQR(dx / dt) / Model.e2;
            real_t boundary_vx = this->params.boundary_values[BOUNDARY_VELOCITY_X][faceId];
            real_t boundary_vy = this->params.boundary_values[BOUNDARY_VELOCITY_Y][faceId];
            real_t phi = this->get_lbm_val(IJKb, IPHI);
            real_t rho = (phi * Model.rho1+ (1-phi) * Model.rho0) ;

            real_t value;
            for (int ipop = 0; ipop < npop; ++ipop) {
                value = dx / dt * this->compute_scal_Ebar(ipop, boundary_vx, boundary_vy) * rho;
                this->compute_boundary_bounceback(tagNS, faceId, IJK, ipop, value);
            }

        } else if (Base::params.boundary_types[BOUNDARY_EQUATION_2][faceId] == BC_POISEUILLE) {
            const real_t dx = this->params.dx;
            const real_t dt = this->params.dt;
            const real_t isize = this->params.isize;
            const real_t jsize = this->params.jsize;
            [[maybe_unused]] const real_t cs2 = SQR(dx / dt) / Model.e2;
            //~ const real_t cs = SQRT(cs2);
            real_t scaling = (faceId == FACE_YMIN or faceId == FACE_YMAX) * 4 * IJK[IX] * (isize - IJK[IX]) / SQR(isize)
                + (faceId == FACE_XMIN or faceId == FACE_XMAX) * 4 * IJK[IY] * (jsize - IJK[IY]) / SQR(jsize);
            real_t boundary_vx = scaling * this->params.boundary_values[BOUNDARY_VELOCITY_X][faceId];
            real_t boundary_vy = scaling * this->params.boundary_values[BOUNDARY_VELOCITY_Y][faceId];
			real_t phi = this->get_lbm_val(IJKb, IPHI);
            real_t rho = (phi * Model.rho1+ (1-phi) * Model.rho0) ;
            real_t value;
            for (int ipop = 0; ipop < npop; ++ipop) {
                value = dx / dt * this->compute_scal_Ebar(ipop, boundary_vx, boundary_vy) * rho;
                this->compute_boundary_bounceback(tagNS, faceId, IJK, ipop, value);
            }
        } else if (Base::params.boundary_types[BOUNDARY_EQUATION_2][faceId] == BC_FREEFLOW) {
			const real_t dx = this->params.dx;
            const real_t dt = this->params.dt;
            [[maybe_unused]] const real_t cs2 = SQR(dx / dt) / Model.e2;
            
            int d=2;
			IVect<dim> IJKm1;
			IJKm1[IX]=IJK[IX]- d * (faceId==FACE_XMAX) + d * (faceId==FACE_XMIN);
			IJKm1[IY]=IJK[IY]- d * (faceId==FACE_YMAX) + d * (faceId==FACE_YMIN);
			if (dim==3)
				IJKm1[IZ]=IJK[IZ]- d * (faceId==FACE_ZMAX) + d * (faceId==FACE_ZMIN);
				
				
            real_t boundary_vx = this->get_lbm_val(IJKm1, IU);
            real_t boundary_vy = this->get_lbm_val(IJKm1, IV);
            real_t phi = this->get_lbm_val(IJKb, IPHI);
            real_t rho = (phi * Model.rho1+ (1-phi) * Model.rho0) ;

            real_t value;
            for (int ipop = 0; ipop < npop; ++ipop) {
                value = dx / dt * this->compute_scal_Ebar(ipop, boundary_vx, boundary_vy) * rho;
                this->compute_boundary_bounceback(tagNS, faceId, IJK, ipop, value);
            }
			//IVect<dim> IJKm2;
			//int d=2;
			//IJKm2[IX]=IJK[IX]- d * (faceId==FACE_XMAX) + d * (faceId==FACE_XMIN);
			//IJKm2[IY]=IJK[IY]- d * (faceId==FACE_YMAX) + d * (faceId==FACE_YMIN);
			//if (dim==3)
				//IJKm2[IZ]=IJK[IZ]- d * (faceId==FACE_ZMAX) + d * (faceId==FACE_ZMIN);
			//d=1;
			//IVect<dim> IJKm1;
			//IJKm1[IX]=IJK[IX]- d * (faceId==FACE_XMAX) + d * (faceId==FACE_XMIN);
			//IJKm1[IY]=IJK[IY]- d * (faceId==FACE_YMAX) + d * (faceId==FACE_YMIN);
			//if (dim==3)
				//IJKm1[IZ]=IJK[IZ]- d * (faceId==FACE_ZMAX) + d * (faceId==FACE_ZMIN);
            //real_t value;
            //for (int ipop = 0; ipop < npop; ++ipop) {

                //value = this->get_f_val(tagNS, IJKm2, ipop);
                //this->set_f_val(tagNS, IJKm1, ipop, value);
            //}
        }
    };

    KOKKOS_INLINE_FUNCTION
    void init_macro(IVect<dim> IJK, RANDOM_POOL::generator_type rand_gen) const
    {

        // get local coordinates
        real_t x, y;
        this->get_coordinates(IJK, x, y);

        // get useful params
        real_t xx = 0.0;
        real_t phi = 0.0;
        // compute phase field
        if (Model.initType == PHASE_FIELD_INIT_VERTICAL)
        {
            xx = x - Model.x0;
        } else if (Model.initType == PHASE_FIELD_INIT_SPHERE)
        {
            xx = (Model.r0 - sqrt(SQR(x - Model.x0) + SQR(y - Model.y0)));
        } else if (Model.initType == PHASE_FIELD_INIT_2SPHERE) 
        {
            real_t d0 = Model.r0-sqrt(SQR(x - Model.x0) + SQR(y - Model.y0));
            real_t d1 = Model.r1-sqrt(SQR(x - Model.x1) + SQR(y - Model.y1));
            xx = FMAX(d0,d1);
        } else if (Model.initType == PHASE_FIELD_INIT_RING) 
        {
            real_t d = sqrt(SQR(x - Model.x0) + SQR(y - Model.y0));
            bool ri = (d > ((Model.r0 + Model.r1) / 2));
            xx = ri * (Model.r0 - d) - (1 - ri) * (Model.r1 - d);
        } else if (Model.initType == PHASE_FIELD_INIT_SQUARE)
        {
            //~ xx = (Model.r0 - FMAX(sqrt(SQR(x-Model.x0)), sqrt(SQR(y-Model.y0)))  );
            xx = FMIN(Model.r0 - sqrt(SQR(x - Model.x0)), Model.r1 - sqrt(SQR(y - Model.y0)));
        } else if (Model.initType == PHASE_FIELD_INIT_MIXTURE)
        {
            xx = x - Model.x0;
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
        this->set_lbm_val(IJK, IU, Model.initVX);
        this->set_lbm_val(IJK, IV, Model.initVY);

    } // end init macro

    KOKKOS_INLINE_FUNCTION
    void update_macro(IVect<dim> IJK) const
    {
        // get useful params
        const real_t dx = this->params.dx;
        const real_t dt = this->params.dt;
        const real_t cs2 = SQR(dx / dt) / Model.e2;
        const real_t c = (dx / dt);

        // compute moments of distribution equations
        real_t moment_phi = 0.0;
        real_t moment_P = 0.0;
        real_t moment_VX = 0.0;
        real_t moment_VY = 0.0;
        for (int ipop = 0; ipop < npop; ++ipop) {
            moment_phi += Base::get_f_val(tagPHI, IJK, ipop);
            moment_P += Base::get_f_val(tagNS, IJK, ipop);
            moment_VX += c * Base::get_f_val(tagNS, IJK, ipop) * E[ipop][IX];
            moment_VY += c * Base::get_f_val(tagNS, IJK, ipop) * E[ipop][IY];
        }

        // store old values of macro fields
        LBMState lbmStatePrev;
        Base::template setupLBMState2<LBMState, COMPONENT_SIZE>(IJK, lbmStatePrev);

        const real_t rhoprev = lbmStatePrev[ID];

        const RVect2 force = Model.force_NS(type, lbmStatePrev);
        const RVect2 forceP = Model.force_P(type, lbmStatePrev);
        const real_t scalP = lbmStatePrev[IU] * forceP[IX] + lbmStatePrev[IV] * forceP[IY];

        const real_t source_phi = Model.S0(type, tagPHI, lbmStatePrev);
        // get source terms and compute new macro fields

        const real_t phi = moment_phi + 0.5 * dt * source_phi;
        const real_t rho = Model.interp_rho(phi);
        // compute NS macro vars
        const real_t P = moment_P + 0.5 * dt * scalP;
        const real_t VX = (moment_VX / cs2 + 0.5 * dt * force[IX]) / rhoprev;
        const real_t VY = (moment_VY / cs2 + 0.5 * dt * force[IY]) / rhoprev;
        // update macro fields
        this->set_lbm_val(IJK, IPHI, phi);
        this->set_lbm_val(IJK, ID, rho);
        this->set_lbm_val(IJK, IP, P);
        this->set_lbm_val(IJK, IU, VX);
        this->set_lbm_val(IJK, IV, VY);
        if (dim == 3)
            this->set_lbm_val(IJK, IV, VY);

    } // end update macro

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
        
        real_t cclabel=Base::get_cc_label(IJK);
        Base::set_lbm_val(IJK, ICC, cclabel);
        
        
    }
    
    


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
            i = (isInBounds)*(phi<scheme.Model.CC_phi_threshold)*i;
            scheme.set_cc_label(IJK, i);
        };

    };


    


    umap map_CCid_to_volume;
    umap map_CCid_to_vx;
    umap map_CCid_to_vy;
    umap map_CCid_to_vz;
    umap map_CCid_to_p;
    umap map_CCid_to_Fx;
    umap map_CCid_to_Fy;
    umap map_CCid_to_Fz;

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
	
			//real_t dx=Model.dx;
            //real_t dv = dim==2 ? dx*dx : dx*dx*dx;
	            
			bool isNotGW= scheme.isNotGhostCell(IJK);
			if (isNotGW) {
	            real_t phi = scheme.get_lbm_val(IJK,IPHI);
	            real_t vx = scheme.get_lbm_val(IJK,IU);
	            real_t vy = scheme.get_lbm_val(IJK,IV);
	            real_t p = scheme.get_lbm_val(IJK,IP);
	            //real_t rho = scheme.Model.rho0;
	            
	            real_t dphidx=scheme.get_lbm_val(IJK,IDPHIDX);
	            real_t dphidy=scheme.get_lbm_val(IJK,IDPHIDY);
	            real_t norm=sqrt(SQR(dphidx)+SQR(dphidy)+NORMAL_EPSILON);
	            
	            //real_t rhoU2 = rho * (1-phi) * (SQR(vx) + SQR(vy))
	            real_t cc = scheme.get_cc_label(IJK);
	            int cci=int(cc);
	
	            scheme.map_CCid_to_volume.insert(cci, (1-phi), scheme.op);
	            scheme.map_CCid_to_vx.insert(cci, ((1-phi) * vx), scheme.op);
	            scheme.map_CCid_to_vy.insert(cci, ((1-phi) * vy), scheme.op);
	            scheme.map_CCid_to_p.insert(cci, ((1-phi) * p), scheme.op);
	            scheme.map_CCid_to_Fx.insert(cci, ( 4 * phi * (1-phi) * p * dphidx/norm), scheme.op);
	            scheme.map_CCid_to_Fy.insert(cci, ( 4 * phi * (1-phi) * p * dphidy/norm), scheme.op);

	        }
        };

    };

    // apply diff
    //~ KOKKOS_INLINE_FUNCTION


    struct FunctorCustomApplyCCsolid
    {
        const LBMScheme<dim,npop, modelType> scheme;
        FunctorCustomApplyCCsolid(const LBMScheme<dim,npop, modelType>& scheme):
            scheme(scheme) {};

        KOKKOS_INLINE_FUNCTION
        void operator ()(const IVect<dim>& IJK) const
        {

            real_t cc = scheme.get_cc_label(IJK);
            int cci=int(cc);

            if (cci>0.5)
            {
				//~ printf("MPI %d, pos %d %d, cc %d\n", scheme.params.myRank, IJK[IX], IJK[IY], cci);
				
                uint32_t id=scheme.map_CCid_to_volume.find(cci);
                if (not(scheme.map_CCid_to_volume.valid_at(id))) throw std::runtime_error("unvalid integral phi");
                real_t Vcc=scheme.map_CCid_to_volume.value_at(id);
                
                id=scheme.map_CCid_to_vx.find(cci);
                if (not(scheme.map_CCid_to_vx.valid_at(id))) throw std::runtime_error("unvalid integral vx");
                real_t rvx=scheme.map_CCid_to_vx.value_at(id);
                
                id=scheme.map_CCid_to_vy.find(cci);
                if (not(scheme.map_CCid_to_vy.valid_at(id))) throw std::runtime_error("unvalid integral vy");
                real_t rvy=scheme.map_CCid_to_vy.value_at(id);
                
                //id=scheme.map_CCid_to_p.find(cci);
                //if (not(scheme.map_CCid_to_p.valid_at(id))) throw std::runtime_error("unvalid integral p");
                //real_t p=scheme.map_CCid_to_p.value_at(id);
                
                id=scheme.map_CCid_to_Fx.find(cci);
                if (not(scheme.map_CCid_to_Fx.valid_at(id))) throw std::runtime_error("unvalid integral F");
                real_t Fx=scheme.map_CCid_to_Fx.value_at(id);
                
                id=scheme.map_CCid_to_Fy.find(cci);
                if (not(scheme.map_CCid_to_Fy.valid_at(id))) throw std::runtime_error("unvalid integral F");
                real_t Fy=scheme.map_CCid_to_Fy.value_at(id);
                
				//real_t vx = rvx/Vcc;
				//real_t vy = rvy/Vcc;
                
				real_t vx = rvx/Vcc + scheme.Model.dt * Fx/Vcc;
				real_t vy = rvy/Vcc + scheme.Model.dt * Fy/Vcc;

                scheme.set_lbm_val(IJK, IU, vx);
                scheme.set_lbm_val(IJK, IV, vy);
                //scheme.set_lbm_val(IJK, IP, p/Vcc);

            }
        };

    };

};

} // end namespace
#endif // LBMSCHEME_NS_PF_H_
