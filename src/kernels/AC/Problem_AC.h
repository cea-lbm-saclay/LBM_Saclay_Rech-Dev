#ifndef PROBLEM_AC_H_
#define PROBLEM_AC_H_

#include "kernels/Collision_operators.h"
#include "kernels/FunctorBoundary.h"
#include "kernels/FunctorInitMacro.h"
#include "kernels/FunctorMacro.h"
#include "kernels/FunctorScheme.h"
#include "kernels/ReducerMax.h"
#include <iomanip>
#include <limits> // for std::numeric_limits
//~ #include <map>
#include <Kokkos_Random.hpp>
#ifdef __CUDA_ARCH__
#include <math_constants.h>

#endif // __CUDA_ARCH__

#include "../../io/io_csv.h"
#include "../ProblemBase.h"
#include "FieldManager.h"
#include "LBMParams.h"
#include "LBMScheme_AC.h"
#include "LBM_Base_Functor.h"

namespace PBM_AC {

template <int dim, int npop, typename modelType>
struct Problem : public ProblemBase<dim, npop> {
    using Base = ProblemBase<dim, npop>;

    using LBMArray = typename LBMBaseFunctor<dim, npop>::LBMArray;
    using LBMArrayHost = typename LBMArray::HostMirror;
    using FArray = typename LBMBaseFunctor<dim, npop>::FArray;
    using FState = typename Kokkos::Array<real_t, npop>;

    using LBMSchemeSpec = LBMScheme<dim, npop, modelType>;

    using MacroKernel = MacroFunctor<dim, npop, LBMSchemeSpec>;
    using MacroInitKernel = MacroInitFunctor<dim, npop, LBMSchemeSpec>;
    using BoundaryFunct = BoundaryFunctor<dim, npop, LBMSchemeSpec>;

    using ReducerMax = MaxReducer<dim, npop, LBMSchemeSpec, EquationTag1>;

    LBMParams params;

    LBMSchemeSpec scheme;

    ConfigMap configMap;

    Problem(ConfigMap& configMap, LBMParams& params)
        : ProblemBase<dim, npop>(configMap, params, index2names::get_id2names())
        , params(params)
        , configMap(configMap)
    {
        Base::nbVar = COMPONENT_SIZE;
        Base::nbEqs = 1;

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
            std::cout << "starting reading init file" << std::endl;
            csvReader reader(params);
            reader.loadFieldFromCSV(Base::Mdata, IPHI, params.initFileName);
            std::cout << "finished reading init file" << std::endl;
        }
        // init Macro
        MacroInitKernel::template apply<TagInitMacro>(params, scheme);
        MacroKernel::template apply<TagUpdateMacroGrads>(params, scheme);
    }
    void init_f() override
    {

        // init distribution functions
        if (params.collisionType1 == BGK) {
            init1eq<EquationTag1, BGKCollider<dim, npop>>();
        } else if (params.collisionType1 == TRT) {
            init1eq<EquationTag1, TRTCollider<dim, npop>>();
        } else if (params.collisionType1 == MRT) {
            init1eq<EquationTag1, MRTCollider<dim, npop>>();
        }
    }

    void update_f() override
    {

        // update phase field distrib
        if (params.collisionType1 == BGK) {
            update1eq<EquationTag1, BGKCollider<dim, npop>>();
        } else if (params.collisionType1 == TRT) {
            update1eq<EquationTag1, TRTCollider<dim, npop>>();
        } else if (params.collisionType1 == MRT) {
            update1eq<EquationTag1, MRTCollider<dim, npop>>();
        }
    }

    void update_m() override
    {

        // update macro fields
        MacroKernel::template apply<TagUpdateMacro>(params, scheme);
        MacroKernel::template apply<TagUpdateMacroGrads>(params, scheme);
    }

    void update_dt() override
    {

        // update timestep
        real_t umax = 0;
        ReducerMax::apply(params, scheme, IU, IV, umax);
        scheme.Model.dtprev = scheme.Model.dt;
        real_t possible_dt = scheme.Model.fMach * scheme.Model.dx / sqrt(3) / umax;
        scheme.Model.dt = possible_dt > 1 ? 1 : possible_dt;
    }

    void update_total_time(real_t m_t) override
    {
        scheme.Model.time = m_t;
    };

    void make_boundaries() override
    {

        BoundaryFunctor<dim, npop, LBMSchemeSpec>::make_all_boundaries(params, scheme);
    }
    void make_periodic_boundaries() override
    {


        this->bcPerMgr.make_boundaries(scheme.f1, BOUNDARY_EQUATION_1);
    }

    void finalize() override
    {

        csvWriter writer(params);
        writer.writeFieldAsCSV(Base::Mdata, IPHI, "./phi.csv");
    }

    real_t get_dt() { return scheme.Model.dt; }

}; // end class ProblemBase

}; // end namespace
#endif // LBM_SCHEME_BASE_H_
