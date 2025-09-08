#ifndef MODELS_GP_MIXT_H_
#define MODELS_GP_MIXT_H_
#include "InitConditionsTypes.h"
#include "Index.h"
#include <real_type.h>
namespace PBM_DIRECTIONAL_SOLIDIFICATION {
// ================================================
//
// Models for mixt formulation of grand potential problems
//
// ================================================

struct ModelParams {

    using LBMState = typename Kokkos::Array<real_t, COMPONENT_SIZE>;
    //~ using localDataTypes;
    static constexpr real_t NORMAL_EPSILON = 1.0e-16;

    void showParams()
    {
        std::cout << "W :    " << W << std::endl;
        std::cout << "mailles par W :    " << W / dx << std::endl;
        

        std::cout << "counter_term :    " << counter_term << std::endl;
        std::cout << "at_current :    " << at_current << std::endl;
        
        std::cout << "lambda :    " << lambda << std::endl;
        
        
        
        std::cout << "init :" << initType << "." << std::endl;
        real_t tau = 0.5 + (e2 * W*W/tauphi * dt / SQR(dx));
        std::cout << "tauPhi :" << tau << std::endl;
        std::cout << "tauSupersat :" << 0.5 + (e2 /gamma* D * dt / SQR(dx)) << std::endl;


    };
    ModelParams() {};
    ModelParams(const ConfigMap& configMap, LBMParams params)
    {
		time=0.0;
        dt = params.dt;
        dx = params.dx;
        e2 = configMap.getFloat("lbm", "e2", 3.0);
        gammaTRT1 = configMap.getFloat("equation1", "lambdaTRT", 1 / 12);
        gammaTRT2 = configMap.getFloat("equation2", "lambdaTRT", 1 / 12);
        
        W = configMap.getFloat("params", "W", 0.005);
        tauphi = configMap.getFloat("params", "tauphi", 1.2);
        counter_term = configMap.getFloat("params", "counter_term", 0.0);
        
        
        lambda = configMap.getFloat("params", "lambda", 0.0);

        D = configMap.getFloat("params", "D", 1.0);
        k = configMap.getFloat("params", "k", 1.0);
        gamma = configMap.getFloat("params", "gamma", 1.0);
        
        Vp = configMap.getFloat("params", "Vp", 0.0);
        lT = configMap.getFloat("params", "lT", 1.0);
        
        at_current = configMap.getFloat("params", "at_current", 0.0);


		
		initU= configMap.getFloat("init", "initU", 0.0);
		x0= configMap.getFloat("init", "x0", 0.0);
		y0= configMap.getFloat("init", "y0", 0.0);
		
		A= configMap.getFloat("init", "A", 0.0);
		kx= configMap.getFloat("init", "kx", 0.0);
		decal= configMap.getFloat("init", "decal", 0.0);


        sign = configMap.getFloat("init", "sign", 1);
        t0 = configMap.getFloat("init", "t0", 1.0);
        initErfc = configMap.getFloat("init", "initErfc", 0.1);
        xi = configMap.getFloat("init", "xi", 0.0);
        phi0_fact = configMap.getFloat("init", "phi0", 1.0);
        //~ init_dphidt = configMap.getFloat("init", "dphidt", 0.0);

        use_sharp_init = configMap.getBool("init", "sharp_init", false);
        use_projected_normal_distance = configMap.getBool("init", "use_projected_normal_distance", false);

        initType = PHASE_FIELD_INIT_UNDEFINED;
        std::string initTypeStr = std::string(configMap.getString("init", "init_type", "unknown"));

        if (initTypeStr == "vertical")
            initType = PHASE_FIELD_INIT_VERTICAL;
        if (initTypeStr == "horizontal")
            initType = PHASE_FIELD_INIT_HORIZONTAL;
        else if (initTypeStr == "cosinus")
            initType = PHASE_FIELD_INIT_COSINUS;


        showParams();
    }

    //! model params

    real_t dx, dt, e2, time; // simulation params
    real_t gammaTRT1, gammaTRT2; // trt params
    real_t W, tauphi, counter_term ; // general params for phi equation
    real_t lambda;
    
    real_t D, gamma, k, at_current; // general params for C equation

	real_t Vp, lT;


    //! init params
    real_t initU;
    real_t x0, y0, z0, r0, sign, t0, initErfc, xi, phi0_fact;
    
    real_t A, kx, decal;
    int initType;
    bool use_sharp_init;
    bool use_projected_normal_distance;

    KOKKOS_INLINE_FUNCTION real_t phi0(real_t x) const { return phi0_fact *  ( tanh(sign * x / SQRT(2)/W)); }

    // interpolation functions

    //~ KOKKOS_INLINE_FUNCTION real_t p(real_t phi) const { return SQR(phi) * (3.0 - 2.0 * phi); }
    //~ KOKKOS_INLINE_FUNCTION real_t p_prime(real_t phi) const { return 6.0 * phi * (1.0 - phi); }
    
    
    //~ KOKKOS_INLINE_FUNCTION real_t fun_g(real_t phi) const { return 8 * SQR(phi) * SQR(1.0 - phi); }
    //~ KOKKOS_INLINE_FUNCTION real_t g_prime(real_t phi) const { return 16.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi); }
    KOKKOS_INLINE_FUNCTION real_t double_well(real_t phi) const { return (SQR(SQR(phi))/4-SQR(phi)/2); }
    KOKKOS_INLINE_FUNCTION real_t double_well_derivative(real_t phi) const { return (phi*phi*phi - phi); }
    //~ KOKKOS_INLINE_FUNCTION real_t h(real_t phi) const { return ((1-phi)/2); }
    //~ KOKKOS_INLINE_FUNCTION real_t hprime(real_t phi) const { return 0.5; }
    KOKKOS_INLINE_FUNCTION real_t q(real_t phi) const { return ((1-phi)/2); }
    KOKKOS_INLINE_FUNCTION real_t qprime(real_t phi) const { return -0.5; }
    
    KOKKOS_INLINE_FUNCTION real_t zeta(real_t phi) const { return ((1+k - (1-k)*phi)/2); }
    KOKKOS_INLINE_FUNCTION real_t zetaprime(real_t phi) const { return ((k-1)/2); }
    KOKKOS_INLINE_FUNCTION real_t zetaprimeprime(real_t phi) const { return 0.0; }
    //~ KOKKOS_INLINE_FUNCTION real_t aa(real_t phi) const { return phi * (3.0 - 2.0 * phi); }
    

    // ================================================
    //
    // functions for the case of 2 quadratic free energies
    //
    // ================================================
    struct TagBasicModel {};



 

    // =======================================================
    // =======================================================
    //
    //		 equation 1 : phase field
    //
    // =======================================================

    // =======================================================
    // factor before temporal derivative
    KOKKOS_INLINE_FUNCTION
    real_t TimeFactor(TagBasicModel type, EquationTag1 tag, const LBMState& lbmState) const
    {
        return  (1-(1-k)*lbmState[ICHI]);
    }
    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(TagBasicModel type, EquationTag1 tag, const LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(TagBasicModel type, EquationTag1 tag, const LBMState& lbmState) const
    {
        return lbmState[IPHI];
    }
    // =======================================================
    // source term phi
    KOKKOS_INLINE_FUNCTION
    real_t S0(TagBasicModel type, EquationTag1 tag, const LBMState& lbmState) const
    {
		const real_t phi =lbmState[IPHI];
		const real_t U =lbmState[ISUPERSAT];
		const real_t chi =lbmState[ICHI];
        real_t source_chem = -lambda * (U +chi) * SQR(1-phi*phi)/tauphi;
        real_t source_noct = -(1.0 - counter_term) * double_well_derivative(phi)/tauphi;
        return (source_chem + source_noct);
    }

    // =======================================================
    // advection terms for phase field
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        S1(TagBasicModel type, EquationTag1 tag, const LBMState& lbmState) const
    {
        const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY])) + NORMAL_EPSILON;
        real_t force_ct = 0.0;
        real_t force_N = 0 * SQR(norm);
        RVect<dim> term;
        term[IX] = (force_ct+force_N) * lbmState[IDPHIDX];
        term[IY] = (force_ct+force_N) * lbmState[IDPHIDY];
        if (dim == 3) {
            term[IZ] = (force_ct+force_N) * lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of phase field equation
    KOKKOS_INLINE_FUNCTION
    real_t tau(TagBasicModel type, EquationTag1 tag, const LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 * W*W /tauphi * dt / SQR(dx));
        return (tau);
    }

    // =======================================================
    // =======================================================
    //
    //		 equation 2 : composition
    //
    // =======================================================


      KOKKOS_INLINE_FUNCTION
    real_t antitrap_factor(const LBMState& lbmState) const
    {
		const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY])) + NORMAL_EPSILON;
        real_t fat = -	at_current* 1/ (2 * SQRT(2)) * W * (1 + (1-k) * lbmState[ISUPERSAT]) *  lbmState[IDPHIDT] / norm;
        return (fat);
    }
    
    
    // =======================================================
    // zero order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M0(TagBasicModel type, EquationTag2 tag, const LBMState& lbmState) const
    {
        return lbmState[ISUPERSAT];
    }

    // =======================================================
    // second order moment of feq
    KOKKOS_INLINE_FUNCTION
    real_t M2(TagBasicModel type, EquationTag2 tag, const LBMState& lbmState) const
    {
        const real_t qv = q(lbmState[IPHI]);
        const real_t zetav = zeta(lbmState[IPHI]);
        const real_t M = gamma * (qv / zetav)  * (lbmState[ISUPERSAT]);
        return M;
    }

    // =======================================================
    // source term C
    KOKKOS_INLINE_FUNCTION
    real_t S0(TagBasicModel type, EquationTag2 tag, const LBMState& lbmState) const
    {
		const real_t U = lbmState[ISUPERSAT];
		const real_t phi = lbmState[IPHI];
		
		const real_t qv = q(phi);
		const real_t qpv = qprime(phi);
		const real_t zv = zeta(phi);
		const real_t zpv = zetaprime(phi);
		const real_t zppv = zetaprimeprime(phi);
		
		const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY])) + NORMAL_EPSILON;
		
		
		real_t Qu = ( 1 + (1-k) * U ) * 0.5 * lbmState[IDPHIDT];
		real_t Sdiv = U * D * ( - qv * zpv / SQR(zv) * lbmState[ILAPLAPHI] + SQR(norm) * (-qpv * zpv/SQR(zv) - qv * zppv / SQR(zv) + qv * 2*SQR(zpv) / (zv*zv*zv)) ) ; // U*D*nabla(q*F)
		
		
		real_t Sscal= antitrap_factor(lbmState) * ( - zpv / (zv*zv)) * SQR(norm); // scal prod between jat and F=nabla(1/zeta(phi))
        return  Sdiv+Sscal + Qu/zv;
    }
    
    
    
    
    // =======================================================
    // advection terms for diffusion equation
    template <int dim>
    KOKKOS_INLINE_FUNCTION
        RVect<dim>
        S1(TagBasicModel type, EquationTag2 tag, const LBMState& lbmState) const
    {
        //~ const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY])) + NORMAL_EPSILON;
        const real_t phi = lbmState[IPHI];
        const real_t U = lbmState[ISUPERSAT];
        
        real_t factor_U = D * ( - 2*q(phi) * zetaprime(phi) / SQR(zeta(phi)) + qprime(phi)/zeta(phi)) * U;
        
        real_t factor_at = antitrap_factor(lbmState);

        RVect<dim> term;
        term[IX] = (factor_U + factor_at) * lbmState[IDPHIDX];
        term[IY] = (factor_U + factor_at) * lbmState[IDPHIDY];
        if (dim == 3) {
            term[IZ] = (factor_U + factor_at) * lbmState[IDPHIDZ];
        }
        return term;
    }

    // =======================================================
    // relaxation coef for LBM scheme of diffusion equation
    KOKKOS_INLINE_FUNCTION
    real_t tau(TagBasicModel type, EquationTag2 tag, const LBMState& lbmState) const
    {
        real_t tau = 0.5 + (e2 /gamma* D * dt / SQR(dx));
        return (tau);
    }

 



 

}; // end struct model

} //end namespace
#endif // MODELS_DISSOL_GP_MIXT_H_
