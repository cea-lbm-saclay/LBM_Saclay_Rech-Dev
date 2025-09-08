#ifndef PROBLEM_GPMIXT_H_
#define PROBLEM_GPMIXT_H_

#include <Kokkos_Random.hpp>
#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>

#endif // __CUDA_ARCH__

#include "FieldManager.h"
#include "LBMParams.h"
#include "LBMScheme_GPMixt.h"
#include "LBM_Base_Functor.h"
#include "../../io/io_csv.h"
namespace PBM_GP_MIXT {
    


template <int dim, int npop, typename modelType>
struct Problem : public ProblemBase<dim, npop> {

    using Base = ProblemBase<dim, npop>;

    using LBMArray = typename LBMBaseFunctor<dim, npop>::LBMArray;
    using LBMArrayHost = typename LBMArray::HostMirror;
    using FArray = typename LBMBaseFunctor<dim, npop>::FArray;
    using FState = typename Kokkos::Array<real_t, npop>;
    static constexpr real_t NORMAL_EPSILON = 1.0e-15;

    using LBMSchemeSpec = LBMScheme<dim, npop, modelType>;
    using MacroKernel = MacroFunctor<dim, npop, LBMSchemeSpec>;
    using MacroInitKernel = MacroInitFunctor<dim, npop, LBMSchemeSpec>;
    //~ using Kernel1 = SchemeFunctor<dim,npop, LBMScheme, EquationTag1, BGKCollider<npop>>;
    //~ using Kernel2 = SchemeFunctor<dim,npop, LBMScheme, EquationTag2, BGKCollider<npop>>;

    LBMParams params;

    LBMSchemeSpec scheme;

    Problem(ConfigMap& configMap, LBMParams& params)
        : ProblemBase<dim, npop>(configMap, params, index2names::get_id2names())
        , params(params)
    {
        Base::nbVar = COMPONENT_SIZE;
        Base::nbEqs = 2;

        const int isize = params.isize;
        const int jsize = params.jsize;
        const int ksize = params.ksize;
        if (dim == 2) {
            Base::Mdata = LBMArray("lbm_data", isize, jsize, Base::nbVar);
            Base::Mdata_h = Kokkos::create_mirror_view(Base::Mdata);

        } else if (dim == 3) {

            Base::Mdata = LBMArray("lbm_data", isize, jsize, ksize, Base::nbVar);
            Base::Mdata_h = Kokkos::create_mirror_view(Base::Mdata);
        }

        ModelParams Model(configMap, params);

        scheme = LBMSchemeSpec(params, Base::Mdata, Model);
        
        scheme.allocate_f(Base::nbEqs);
    };

    template <typename EquationTag, typename Collider>
    void update1eq()
    {
        using Kernel = SchemeFunctor<dim, npop, LBMSchemeSpec, EquationTag, Collider>;
        Kernel::template apply<TagUpdate>(params, scheme);
        EquationTag tag;
        scheme.swap_distribution(tag);
    }

    template <typename EquationTag, typename Collider>
    void init1eq()
    {
        using Kernel = SchemeFunctor<dim, npop, LBMSchemeSpec, EquationTag, Collider>;
        Kernel::template apply<TagInit>(params, scheme);
    }

    void init_m() override
    {

        if (params.init_str == "data") {
            csvReader reader(params);
            reader.loadFieldFromCSV(Base::Mdata, IPHI, params.initFileName);
        }

        // init Macro
        MacroInitKernel::template apply<TagInitMacro>(params, scheme);
        
        
        bool inf_diff_connected_components = Base::configMap.getBool("params", "infinite_diff_connected_components", true);
        
        if (inf_diff_connected_components) 
        {

            Base::template compute_connected_components<LBMSchemeSpec, typename LBMSchemeSpec::FunctorCustomInitCClabels>(scheme);
            

        }
        
        MacroKernel::template apply<TagUpdateMacroGrads>(params, scheme);
    }

    void init_f() override
    {

        // init distribution functions
        if (params.collisionType1 == BGK) {
            init1eq<EquationTag1, BGKCollider<dim, npop>>();
        } else if (params.collisionType1 == TRT) {
            init1eq<EquationTag1, TRTCollider<dim, npop>>();
        }

        if (params.collisionType2 == BGK) {
            init1eq<EquationTag2, BGKCollider<dim, npop>>();
        } else if (params.collisionType2 == TRT) {
            init1eq<EquationTag2, TRTCollider<dim, npop>>();
        }
    }

    void update_f() override
    {
        // update phase field distrib
        if (params.collisionType1 == BGK) {
            update1eq<EquationTag1, BGKCollider<dim, npop>>();
        } else if (params.collisionType1 == TRT) {
            update1eq<EquationTag1, TRTCollider<dim, npop>>();
        }

        // update concentration distrib
        if (params.collisionType2 == BGK) {
            update1eq<EquationTag2, BGKCollider<dim, npop>>();
        } else if (params.collisionType2 == TRT) {
            update1eq<EquationTag2, TRTCollider<dim, npop>>();
        }
    }

    void update_m() override
    {

        // update macro fields
        MacroKernel::template apply<TagUpdateMacro>(params, scheme);
        MacroKernel::template apply<TagUpdateMacroGrads>(params, scheme);
        
        bool check_conservation = Base::configMap.getBool("params", "check_conservation", false);
        real_t sum_C=0;
        if (check_conservation)
        {
            ReducerSum<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorSumC>::apply(params, scheme, sum_C);
        }
        
        
        bool output_CC_info = Base::configMap.getBool("params", "output_CC_info", false);
        
        bool inf_diff_all_liquid = Base::configMap.getBool("params", "infinite_diff_liquid", false);
        
        if (inf_diff_all_liquid)
        {
            
            // Step 1: compute phi and phiCl integrals for each cc
            real_t sum_phi=0;
            ReducerSum<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorSumPhi>::apply(params, scheme, sum_phi);
            real_t sum_phiCl=0;
            ReducerSum<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorSumPhiCl>::apply(params, scheme, sum_phiCl);
            
            
            scheme.nltot=sum_phiCl;
            scheme.vl=sum_phi;
            // Step 2: update composition values
            CustomFunctor<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorCustomUpdateC>::apply(params, scheme);
            
            // Step 3: update distrib to correspond to new values
            init_f();
            
        
        }
        
        
        bool inf_diff_connected_components = Base::configMap.getBool("params", "infinite_diff_connected_components", false);
        
        
        if (inf_diff_connected_components) 
        {
            // reset the maps
            scheme.map_CCid_to_phi.clear();
            scheme.map_CCid_to_phiCl.clear();
            
            // step 1: determine connected components

            Base::template compute_connected_components<LBMSchemeSpec, typename LBMSchemeSpec::FunctorCustomInitCClabels>(scheme);
            
            // Step 2: compute phi and phiCl integrals for each cc
            CustomFunctor<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorCustomComputeIntegrals>::apply(params, scheme);
            
            // Step 2b: if using mpi, regroup the integrals over different parts of the domain
            #ifdef USE_MPI
                Base::mpi_sum_map(scheme.map_CCid_to_phi);
                Base::mpi_sum_map(scheme.map_CCid_to_phiCl);
            #endif
            
            // Step 3: update composition values
            CustomFunctor<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorCustomApplyCCdiff>::apply(params, scheme);
            
            // Step 4: update distrib to correspond to new values
            init_f();
            

            if (output_CC_info)
            {
                std::cout<<"countcc    : "<<scheme.map_CCid_to_phi.size()<<std::endl;
                
            }
            
            
        
        }
        
        if (check_conservation)
        {
            real_t sum_C2=0;
            ReducerSum<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorSumC>::apply(params, scheme, sum_C2);
            std::cout<<"c    : "<<sum_C<<std::endl;
            std::cout<<"cdiff: "<<sum_C2<<std::endl;
        }
        
    } // end update_m
    

    void make_boundaries() override
    {

       BoundaryFunctor<dim, npop, LBMSchemeSpec>::make_all_boundaries(params, scheme);

    }

    void make_periodic_boundaries() override
    {


        Base::bcPerMgr.make_boundaries(scheme.f1, BOUNDARY_EQUATION_1);
        Base::bcPerMgr.make_boundaries(scheme.f2, BOUNDARY_EQUATION_2);
    }
    void hook_before_output(int step) override
    {
        // copy cc labels to lbm_data array for output 
        CustomFunctor<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorCustom_Copy_CClabels_for_output>::apply(params, scheme);

    }
    void update_dt() override
    {
    }

    void update_total_time(real_t m_t) override
    {
        scheme.Model.time = m_t;
    };

    real_t get_dt()
    {
        return scheme.Model.dt;
    }
// int get_nbEqs() {return Base::nbEqs;}


}; // end class ProblemBase

} // end namespace

#endif // LBM_SCHEME_BASE_H_
