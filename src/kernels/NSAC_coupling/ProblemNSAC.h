/**
 * \file ProblemNSAC.h
 * \brief Problem class where the scheme is chosen, and functors are called
 *
 */

#ifndef PROBLEM_NS_PF_H_
#define PROBLEM_NS_PF_H_

#include "LBMScheme_NS_AC.h"


namespace PBM_NSAC {

template <int dim, int npop, typename modelType>
struct Problem : public ProblemBase<dim, npop> {

    using Base = ProblemBase<dim, npop>;

    using LBMArray = typename LBMBaseFunctor<dim, npop>::LBMArray;
    using LBMArrayHost = typename LBMArray::HostMirror;
    using FArray = typename LBMBaseFunctor<dim, npop>::FArray;
    using FState = typename Kokkos::Array<real_t, npop>;

    using LBMSchemeSpec = LBMScheme<dim, npop, modelType>;
    using Kernel1 = SchemeFunctor<dim, npop, LBMSchemeSpec, EquationTag1, BGKCollider<dim, npop>>;
    using Kernel2 = SchemeFunctor<dim, npop, LBMSchemeSpec, EquationTag2, BGKCollider<dim, npop>>;
    using MacroKernel = MacroFunctor<dim, npop, LBMSchemeSpec>;
    using MacroInitKernel = MacroInitFunctor<dim, npop, LBMSchemeSpec>;
    using BoundaryKernel = BoundaryFunctor<dim, npop, LBMSchemeSpec>;

    using ReducerMax = MaxReducer<dim, npop, LBMSchemeSpec, EquationTag2>;

    LBMParams params;

    LBMSchemeSpec scheme;

    Problem(ConfigMap& configMap, LBMParams& params)
        : ProblemBase<dim, npop>(configMap, params, index2names::get_id2names())
        , params(params)
    {
        //~ if (params.communicator->getRank()==0){

        //~ std::unordered_map<std::string, int> sizesList = {
        //~ { "TRT collider", int(sizeof(BGKCollider<npop>)) },
        //~ { "BGK collider", int(sizeof(TRTCollider<dim,npop>)) },
        //~ { "params", int(sizeof(LBMParams)) },
        //~ { "model", int(sizeof(ModelParams)) },
        //~ { "schemeBase", int(sizeof(LBMSchemeBase<dim,npop>)) },
        //~ { "scheme", int(sizeof(LBMSchemeSpec)) },
        //~ { "bdy functor", int(sizeof(BoundaryFunct)) },
        //~ { "macro functor", int(sizeof(MacroKernel)) },
        //~ { "scheme functor", int(sizeof(Kernel1)) },
        //~ { "problemBase", int(sizeof(ProblemBase<dim,npop>)) },

        //~ };
        //~ std::cout<<"Structure sizes in bits"<<std::endl;
        //~ std::cout<<std::left;
        //~ for(auto iterator = sizesList.begin(); iterator != sizesList.end(); iterator++){
        //~ int collen=FMAX(std::to_string(iterator->second).length(),iterator->first.length());
        //~ std::cout<<std::setw(collen)<<iterator->first<<'|';
        //~ }
        //~ std::cout<<std::endl<<std::left;
        //~ for(auto iterator = sizesList.begin(); iterator != sizesList.end(); iterator++){
        //~ int collen=FMAX(std::to_string(iterator->second).length(),iterator->first.length());
        //~ std::cout<<std::setw(collen)<<iterator->second<<'|';
        //~ }
        //~ std::cout<<std::endl;

        //~ }
        Base::nbVar = COMPONENT_SIZE;
        Base::nbEqs = 2;

        const int isize = params.isize;
        const int jsize = params.jsize;
        const int ksize = params.ksize;
        if (dim == 2) {
            Base::Mdata = LBMArray("Mdata", isize, jsize, Base::nbVar);
            Base::Mdata_h = Kokkos::create_mirror_view(Base::Mdata);

        } else if (dim == 3) {

            Base::Mdata = LBMArray("Mdata", isize, jsize, ksize, Base::nbVar);
            Base::Mdata_h = Kokkos::create_mirror_view(Base::Mdata);
        }

        scheme = LBMSchemeSpec(configMap, params, Base::Mdata);
        
        scheme.allocate_f(Base::nbEqs);
    };

    void init_m() override
    {

        //~ std::string init_str = std::string(configMap.getString("init", "init_type", "unknown"));
        //~ if (init_str == "data") {loadImageData(data);}

        // init Macro
        MacroInitKernel::template apply<TagInitMacro>(params, scheme);
        
        bool solid_connected_components = Base::configMap.getBool("params", "solid_connected_components", false);
        
        if (solid_connected_components) 
        {
			

	        Base::template compute_connected_components<LBMSchemeSpec, typename LBMSchemeSpec::FunctorCustomInitCClabels>(scheme);
	       

	        
		}
		
		
        MacroKernel::template apply<TagUpdateMacroGrads>(params, scheme);
    }
    void init_f() override
    {

        // init distribution functions
        Kernel1::template apply<TagInit>(params, scheme);
        Kernel2::template apply<TagInit>(params, scheme);
    }

    void update_f() override
    {

        // update phase field distrib
        Kernel1::template apply<TagUpdate>(params, scheme);
        EquationTag1 tag1;
        scheme.swap_distribution(tag1);

        // update fluid distrib
        Kernel2::template apply<TagUpdate>(params, scheme);
        EquationTag2 tag2;
        scheme.swap_distribution(tag2);
    }

    void update_m() override
    {

        // update macro fields
        MacroKernel::template apply<TagUpdateMacro>(params, scheme);
        
        
        bool solid_connected_components = Base::configMap.getBool("params", "solid_connected_components", false);
        
        if (solid_connected_components) 
        {
			
			// reset the maps
			scheme.map_CCid_to_volume.clear();
			scheme.map_CCid_to_vx.clear();
			scheme.map_CCid_to_vy.clear();
			scheme.map_CCid_to_vz.clear();
			scheme.map_CCid_to_p.clear();
			scheme.map_CCid_to_Fx.clear();
			scheme.map_CCid_to_Fy.clear();
			scheme.map_CCid_to_Fz.clear();
			
			
			// step 1: determine connected components

	        Base::template compute_connected_components<LBMSchemeSpec, typename LBMSchemeSpec::FunctorCustomInitCClabels>(scheme);
	        
	        //~ printf("MPI %d, compute integrals\n", params.myRank);
			
			// Step 2: compute phi and phiCl integrals for each cc
			CustomFunctor<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorCustomComputeIntegrals>::apply(params, scheme);
			
			// Step 2b: if using mpi, regroup the integrals over different parts of the domain
			#ifdef USE_MPI
				//~ printf("MPI %d, mpi join sums\n", params.myRank);
				Base::mpi_sum_map(scheme.map_CCid_to_volume);
				Base::mpi_sum_map(scheme.map_CCid_to_vx);
				Base::mpi_sum_map(scheme.map_CCid_to_vy);
			#endif
			// Step 3: update composition values
			//~ printf("MPI %d, update compo\n", params.myRank);
			CustomFunctor<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorCustomApplyCCsolid>::apply(params, scheme);
			
			// Step 4: update distrib to correspond to new values
			//~ printf("MPI %d, initf\n", params.myRank);
			init_f();
			
			
			
			//~ for (uint32_t i=0;i<(scheme.map_CCid_to_phi.capacity());i++)
			//~ {
				//~ if( scheme.map_CCid_to_phi.valid_at(i) ) {
			        //~ auto key   = scheme.map_CCid_to_phi.key_at(i);
			        //~ auto value = scheme.map_CCid_to_phi.value_at(i);
			        //~ std::cout<<"Vcc    : "<<value<<" at " << key <<std::endl;
			    //~ }
				
			//~ }

			

		
		}
		
		MacroKernel::template apply<TagUpdateMacroGrads>(params, scheme);
		
		
    }

    void update_dt() override
    {

        // update timestep
        real_t umax = 0;
        ReducerMax::apply(params, scheme, IU, IV, umax);
        //~ scheme.Model.dtprev = scheme.Model.dt;
        //~ umax = sqrt(-scheme.Model.gy*scheme.Model.time);

        //~ real_t possible_dt = scheme.Model.fMach * scheme.Model.dx / sqrt(3) / sqrt(umax*umax/scheme.Model.L-scheme.Model.gy/scheme.Model.L)/scheme.Model.W;

        //~ scheme.Model.dt = possible_dt ;
    }

    void update_total_time(real_t m_t) override
    {
        scheme.Model.time = m_t;
    };

    real_t get_dt() { return scheme.Model.dt; }

    void make_boundaries() override
    {

        BoundaryFunctor<dim, npop, LBMSchemeSpec>::make_all_boundaries(params, scheme);

        this->bcPerMgr.make_boundaries(scheme.f1, BOUNDARY_EQUATION_1);
        this->bcPerMgr.make_boundaries(scheme.f2, BOUNDARY_EQUATION_2);
    }

}; // end class pbm ns ac

}; //end namespace
#endif // LBM_SCHEME_BASE_H_
