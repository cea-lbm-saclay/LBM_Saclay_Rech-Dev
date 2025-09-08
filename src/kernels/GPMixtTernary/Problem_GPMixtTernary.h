/**
 * \file Problem_GPMixtTernary.h
 *
 */

#ifndef PROBLEM_GPMIXT_TERNARY_H_
#define PROBLEM_GPMIXT_TERNARY_H_

#include <Kokkos_Random.hpp>
#include <fstream>
#include <limits> // for std::numeric_limits
#include <vector>

#ifdef __CUDA_ARCH__
#include <math_constants.h>

#endif // __CUDA_ARCH__

#include "FieldManager.h"
#include "LBMParams.h"
#include "LBMScheme_GPMixt_Ternary.h"
#include "LBM_Base_Functor.h"

#include "../../io/io_csv.h"

namespace PBM_GP_MIXT_TERNARY
{

template<int dim, int npop, typename modelType>
struct Problem : public ProblemBase<dim, npop>
{
    using Base = ProblemBase<dim, npop>;

    using id2index_t = Kokkos::Array<int, COMPONENT_SIZE>;
    using LBMArray = typename LBMBaseFunctor<dim, npop>::LBMArray;
    using LBMArrayHost = typename LBMArray::HostMirror;
    using FArray = typename LBMBaseFunctor<dim, npop>::FArray;
    using FState = typename Kokkos::Array<real_t, npop>;
    static constexpr real_t NORMAL_EPSILON = 1.0e-15;

    //~ using LBMSchemeSpec = LBMScheme<dim, npop, modelType>;
    typedef LBMScheme<dim, npop, modelType> LBMSchemeSpec;
    using MacroKernel = MacroFunctor<dim, npop, LBMSchemeSpec>;
    using MacroInitKernel = MacroInitFunctor<dim, npop, LBMSchemeSpec>;

    LBMParams params;

    LBMSchemeSpec scheme;

    Problem(ConfigMap& configMap, LBMParams& params)
      : ProblemBase<dim, npop>(configMap, params, index2names::get_id2names())
      , params(params)
    {
        Base::nbVar = COMPONENT_SIZE;
        Base::nbEqs = 3;

        const int isize = params.isize;
        const int jsize = params.jsize;
        const int ksize = params.ksize;
        if (dim == 2)
        {
            Base::Mdata = LBMArray("lbm_data", isize, jsize, Base::nbVar);
            Base::Mdata_h = Kokkos::create_mirror_view(Base::Mdata);
        }
        else if (dim == 3)
        {

            Base::Mdata = LBMArray("lbm_data", isize, jsize, ksize, Base::nbVar);
            Base::Mdata_h = Kokkos::create_mirror_view(Base::Mdata);
        }

        ModelParams Model(configMap, params);

        scheme = LBMSchemeSpec(params, Base::Mdata, Model);
        
        scheme.allocate_f(Base::nbEqs);
    };

    template<typename EquationTag, typename Collider>
    void update1eq()
    {
        using Kernel = SchemeFunctor<dim, npop, LBMSchemeSpec, EquationTag, Collider>;
        Kernel::template apply<TagUpdate>(params, scheme);
        EquationTag tag;
        scheme.swap_distribution(tag);
    }

    template<typename EquationTag, typename Collider>
    void init1eq()
    {
        using Kernel = SchemeFunctor<dim, npop, LBMSchemeSpec, EquationTag, Collider>;
        Kernel::template apply<TagInit>(params, scheme);
    }

    void init_m() override
    {
        if (scheme.Model.read_data_phi)
        {
            csvReader reader(params);
            reader.loadFieldFromCSV(Base::Mdata, IPHI, scheme.Model.file_data_phi);
        }

        // init Macro
        MacroInitKernel::template apply<TagInitMacro>(params, scheme);

        if (scheme.Model.use_connected_components) // this is only done to get correct output after first iteration
        {

            // step 1: determine connected components
            Base::template compute_connected_components<LBMSchemeSpec, typename LBMSchemeSpec::FunctorCustomInitCClabels>(scheme);
            // Step 2: compute phi and phiCl integrals for each cc
            CustomFunctor<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorCustomComputeIntegrals>::apply(params, scheme);
            // Step 2b: if using mpi, regroup the integrals over different parts of the domain
#ifdef USE_MPI
            //~ printf("MPI %d, mpi join sums\n", params.myRank);
            Base::mpi_sum_map(scheme.map_CCid_to_phi);
            Base::mpi_sum_map(scheme.map_CCid_to_phiClA);
            Base::mpi_sum_map(scheme.map_CCid_to_phiClB);
#endif

            // Apply virtual volume to the bulk liquid phase
            if (scheme.Model.apply_virtual_volume)
            {
                apply_virtual_volume();
            }
        }

        MacroKernel::template apply<TagUpdateMacroGrads>(params, scheme);
    }

    void init_f() override
    {
        // init distribution functions
        if (params.collisionType1 == BGK)
        {
            init1eq<EquationTag1, BGKCollider<dim, npop>>();
        }
        else if (params.collisionType1 == TRT)
        {
            init1eq<EquationTag1, TRTCollider<dim, npop>>();
        }

        // init concentration distrib A
        if (params.collisionType2 == BGK)
        {
            init1eq<EquationTag2, BGKCollider<dim, npop>>();
        }
        else if (params.collisionType2 == TRT)
        {
            init1eq<EquationTag2, TRTCollider<dim, npop>>();
        }

        // init concentration distrib B
        if (params.collisionType3 == BGK)
        {
            init1eq<EquationTag3, BGKCollider<dim, npop>>();
        }
        else if (params.collisionType3 == TRT)
        {
            init1eq<EquationTag3, TRTCollider<dim, npop>>();
        }
    }

    void update_f() override
    {

        if (params.collisionType1 == BGK)
        {
            update1eq<EquationTag1, BGKCollider<dim, npop>>();
        }
        else if (params.collisionType1 == TRT)
        {
            update1eq<EquationTag1, TRTCollider<dim, npop>>();
        }

        // update concentration distrib A
        if (params.collisionType2 == BGK)
        {
            update1eq<EquationTag2, BGKCollider<dim, npop>>();
        }
        else if (params.collisionType2 == TRT)
        {
            update1eq<EquationTag2, TRTCollider<dim, npop>>();
        }

        // update concentration distrib B
        if (params.collisionType3 == BGK)
        {
            update1eq<EquationTag3, BGKCollider<dim, npop>>();
        }
        else if (params.collisionType3 == TRT)
        {
            update1eq<EquationTag3, TRTCollider<dim, npop>>();
        }
    }

    void update_m() override
    {

        // update macro fields
        MacroKernel::template apply<TagUpdateMacro>(params, scheme);

        bool check_conservation = Base::configMap.getBool("params", "check_conservation", false);
        real_t sum_CA = 0;
        real_t sum_CB = 0;
        if (check_conservation)
        {
            ReducerSum<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorSumCA>::apply(params, scheme, sum_CA);
            ReducerSum<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorSumCB>::apply(params, scheme, sum_CB);
        }

        bool inf_diff_all_liquid = Base::configMap.getBool("params", "infinite_diff_liquid", false);

        if (inf_diff_all_liquid)
        {

            // Step 1: compute phi and phiCl integrals for each cc
            real_t sum_phi = 0;
            ReducerSum<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorSumPhi>::apply(params, scheme, sum_phi);
            real_t sum_phiClA = 0;
            ReducerSum<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorSumphiClA>::apply(params, scheme, sum_phiClA);
            real_t sum_phiClB = 0;
            ReducerSum<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorSumphiClB>::apply(params, scheme, sum_phiClB);


            if (scheme.Model.apply_virtual_volume)
            {
                sum_phiClA+=scheme.Model.cA_in_virtual_volume;
                sum_phiClB+=scheme.Model.cB_in_virtual_volume;
                sum_phi+=scheme.Model.virtual_volume;
                
                scheme.Model.cA_in_virtual_volume = sum_phiClA * scheme.Model.virtual_volume / sum_phi;
            }
            scheme.nltotA = sum_phiClA;
            scheme.nltotB = sum_phiClB;
            scheme.vl = sum_phi;
            // Step 2: update composition values
            CustomFunctor<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorCustomUpdateC>::apply(params, scheme);

            // Step 3: update distrib to correspond to new values
            init_f();
            //~ make_periodic_boundaries();
        }

        if (scheme.Model.use_connected_components)
        {

            // reset the maps
            scheme.CCid_vv.clear();
            scheme.map_CCid_to_phi.clear();
            scheme.map_CCid_to_phiClA.clear();
            scheme.map_CCid_to_phiClB.clear();

            // step 1: determine connected components

            Base::template compute_connected_components<LBMSchemeSpec, typename LBMSchemeSpec::FunctorCustomInitCClabels>(scheme);

            //~ printf("MPI %d, compute integrals\n", params.myRank);

            // Step 2: compute phi and phiCl integrals for each cc
            CustomFunctor<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorCustomComputeIntegrals>::apply(params, scheme);

            // Step 2b: if using mpi, regroup the integrals over different parts of the domain
#ifdef USE_MPI
            //~ printf("MPI %d, mpi join sums\n", params.myRank);
            Base::mpi_sum_map(scheme.map_CCid_to_phi);
            Base::mpi_sum_map(scheme.map_CCid_to_phiClA);
            Base::mpi_sum_map(scheme.map_CCid_to_phiClB);
#endif

            // Apply virtual volume to the bulk liquid phase
            if (scheme.Model.apply_virtual_volume)
            {
                apply_virtual_volume();
            }
            // Step 3: update composition values
            //~ printf("MPI %d, update compo\n", params.myRank);
            CustomFunctor<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorCustomApplyCCdiff>::apply(params, scheme);

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

        if (check_conservation)
        {
            real_t sum_CA2 = 0;
            real_t sum_CB2 = 0;
            ReducerSum<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorSumCA>::apply(params, scheme, sum_CA2);
            ReducerSum<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorSumCB>::apply(params, scheme, sum_CB2);
            if (params.myRank == 0)
            {
                printf("MPI rank : %d | cA    :%10.10f\n", params.myRank, sum_CA);
                printf("MPI rank : %d | cAdiff:%10.10f\n", params.myRank, sum_CA2);
                printf("MPI rank : %d | cB    :%10.10f\n", params.myRank, sum_CB);
                printf("MPI rank : %d | cBdiff:%10.10f\n", params.myRank, sum_CB2);
            }
        }

        MacroKernel::template apply<TagUpdateMacroGrads>(params, scheme);
    } // end udpate m

    void make_boundaries() override
    {

        BoundaryFunctor<dim, npop, LBMSchemeSpec>::make_all_boundaries(params,
                                                                       scheme);
    }

    void make_periodic_boundaries() override
    {

        this->bcPerMgr.make_boundaries(scheme.f1, BOUNDARY_EQUATION_1);
        this->bcPerMgr.make_boundaries(scheme.f2, BOUNDARY_EQUATION_2);
        this->bcPerMgr.make_boundaries(scheme.f3, BOUNDARY_EQUATION_3);
    }
    void hook_before_output(int step) override
    {
        
        if (scheme.Model.apply_virtual_volume)
        {
            std::cout<< "cA_in_virtual_volume : " << scheme.Model.cA_in_virtual_volume << std::endl; 
            std::cout<< "cB_in_virtual_volume : " << scheme.Model.cB_in_virtual_volume << std::endl; 
        }

        if (scheme.Model.use_connected_components)
        {
            // copy cc labels to lbm_data array for output
            CustomFunctor<dim, npop, LBMSchemeSpec, typename LBMSchemeSpec::FunctorCustom_Copy_CClabels_for_output>::apply(params, scheme);

            // output cc data: store maps into vector with all data then write as csv

            // copy maps to host space
            umap_h map_CCid_to_phi_h(KOKKOS_MAP_DEFAULT_CAPACITY);
            umap_h map_CCid_to_phiClA_h(KOKKOS_MAP_DEFAULT_CAPACITY);
            umap_h map_CCid_to_phiClB_h(KOKKOS_MAP_DEFAULT_CAPACITY);
            Kokkos::deep_copy(map_CCid_to_phi_h, scheme.map_CCid_to_phi);
            Kokkos::deep_copy(map_CCid_to_phiClA_h, scheme.map_CCid_to_phiClA);
            Kokkos::deep_copy(map_CCid_to_phiClB_h, scheme.map_CCid_to_phiClB);

            std::vector<std::array<real_t, 4>> ccdata(0);

            const uint32_t n = map_CCid_to_phi_h.capacity();
            //! get the labels from the kokkos map
            for (uint32_t i = 0; i < n; i++)
            {

                if (map_CCid_to_phi_h.valid_at(i))
                {
                    int cclabel = map_CCid_to_phi_h.key_at(i);
                    //~ std::cout << i <<"  "<< cclabel << std::endl;
                    if (cclabel > 0)
                    {
                        real_t phi = map_CCid_to_phi_h.value_at(i);
                        real_t phiClA = map_CCid_to_phiClA_h.value_at(map_CCid_to_phiClA_h.find(cclabel));
                        real_t phiClB = map_CCid_to_phiClB_h.value_at(map_CCid_to_phiClB_h.find(cclabel));
                        ccdata.push_back({ (1.0 * cclabel), phi, phiClA, phiClB });
                    }
                }
            }
            std::string outputDir = Base::configMap.getString("output", "outputDir", "./");
            std::string outputPrefix = Base::configMap.getString("output", "outputPrefix", "output");
            auto outNum = lbm_saclay::format_index(step, 7);

            std::string filename = outputDir + "/" + outputPrefix + "_ccdata_" + outNum + ".csv";

            std::ofstream myFile;
            myFile.open(filename);
            myFile.precision(10);

            std::string sep = ",";

            myFile << "label" << sep << "volume" << sep << "ClA" << sep << "ClB" << std::endl;
            for (auto it = ccdata.begin(); it != ccdata.end(); ++it)
            {
                // for (auto val : *it)
                // {
                // myFile << val << sep;
                // }
                for (int j = 0; j < 4; j++)
                {
                    myFile << (*it)[j];
                    if (j < 3)
                    {
                        myFile << sep;
                    }
                }
                myFile << std::endl;
            }

            // Close file
            myFile.close();

        } // end if scheme.Model.use_connected_components
    }

    void apply_virtual_volume()
    {

        // copy maps to host space
        umap_h map_CCid_to_phi_h(KOKKOS_MAP_DEFAULT_CAPACITY);
        umap_h map_CCid_to_phiClA_h(KOKKOS_MAP_DEFAULT_CAPACITY);
        umap_h map_CCid_to_phiClB_h(KOKKOS_MAP_DEFAULT_CAPACITY);
        uset_h CCid_vv_h(KOKKOS_MAP_DEFAULT_CAPACITY);
        Kokkos::deep_copy(map_CCid_to_phi_h, scheme.map_CCid_to_phi);
        Kokkos::deep_copy(map_CCid_to_phiClA_h, scheme.map_CCid_to_phiClA);
        Kokkos::deep_copy(map_CCid_to_phiClB_h, scheme.map_CCid_to_phiClB);
        Kokkos::deep_copy(CCid_vv_h, scheme.CCid_vv);

        const uint32_t n = CCid_vv_h.capacity();
        for (uint32_t i = 0; i < n; i++)
        {
            if (CCid_vv_h.valid_at(i))
            {
                int cclabel = CCid_vv_h.key_at(i);
                scheme.Model.cclabel_connected_to_virtual_volume=cclabel;
                
            }
        }
        if (scheme.Model.print_cc_trace)
        {
            std::cout << "label of virtual volume : " << scheme.Model.cclabel_connected_to_virtual_volume << std::endl;
        }

        int cc_label_vv=scheme.Model.cclabel_connected_to_virtual_volume;
        if (cc_label_vv > 0)
        {
            real_t v_real = 0;
            real_t cA_real = 0;
            real_t cB_real = 0;
        
            v_real = map_CCid_to_phi_h.value_at(map_CCid_to_phi_h.find(cc_label_vv));
            cA_real = map_CCid_to_phiClA_h.value_at(map_CCid_to_phiClA_h.find(cc_label_vv));
            cB_real = map_CCid_to_phiClB_h.value_at(map_CCid_to_phiClB_h.find(cc_label_vv));
            

            AtomicAddHost op;
            map_CCid_to_phi_h.insert(cc_label_vv, scheme.Model.virtual_volume, op);
            map_CCid_to_phiClA_h.insert(cc_label_vv, scheme.Model.cA_in_virtual_volume, op);
            map_CCid_to_phiClB_h.insert(cc_label_vv, scheme.Model.cB_in_virtual_volume, op);

    
            real_t vv_factor = scheme.Model.virtual_volume / (v_real + scheme.Model.virtual_volume);
            scheme.Model.cA_in_virtual_volume = (cA_real + scheme.Model.cA_in_virtual_volume) * vv_factor;
            scheme.Model.cB_in_virtual_volume = (cB_real + scheme.Model.cB_in_virtual_volume) * vv_factor;
                
        }
        


        // scheme.map_CCid_to_phi.insert(scheme.Model.cclabel_connected_to_virtual_volume, scheme.Model.virtual_volume, scheme.op)


        // copy back to device
        Kokkos::deep_copy(scheme.map_CCid_to_phi, map_CCid_to_phi_h);
        Kokkos::deep_copy(scheme.map_CCid_to_phiClA, map_CCid_to_phiClA_h);
        Kokkos::deep_copy(scheme.map_CCid_to_phiClB, map_CCid_to_phiClB_h);
    }



    void update_dt() override {}

    void update_total_time(real_t m_t) override { scheme.Model.time = m_t; };

    real_t get_dt() { return scheme.Model.dt; }

}; // end class Problem mixt ternary

} // namespace PBM_GP_MIXT_TERNARY

#endif // LBM_SCHEME_BASE_H_
