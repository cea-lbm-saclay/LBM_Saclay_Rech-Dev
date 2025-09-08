#ifndef PROBLEM_NS_PF_F_H_
#define PROBLEM_NS_PF_F_H_

#include <iomanip>
#include <limits> // for std::numeric_limits
//~ #include <map>
#include <Kokkos_Random.hpp>
#ifdef __CUDA_ARCH__
#include <math_constants.h>

#endif // __CUDA_ARCH__

#include "FieldManager.h"
#include "LBMParams.h"
#include "LBMScheme_NS_AC_Fakhari.h"
#include "LBM_Base_Functor.h"

namespace PBM_NS_AC_Fakhari {

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

    using ReducerMax = MaxReducer<dim, npop, LBMSchemeSpec, EquationTag2>;

    LBMParams params;

    LBMSchemeSpec scheme;
    ConfigMap configMap;

    Problem(ConfigMap& configMap, LBMParams& params)
        : ProblemBase<dim, npop>(configMap, params, index2names::get_id2names())
        , params(params)
        , configMap(configMap)
    {
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
        if (params.collisionType1 == BGK) {
            std::cout << "using bgk collision for phi equation" << std::endl;
        } else if (params.collisionType1 == TRT) {
            std::cout << "using trt collision for phi equation" << std::endl;
        } else if (params.collisionType1 == MRT) {
            std::cout << "using mrt collision for phi equation" << std::endl;
        }

        if (params.collisionType2 == BGK) {
            std::cout << "using bgk collision for ca equation" << std::endl;
        } else if (params.collisionType2 == TRT) {
            std::cout << "using trt collision for ca equation" << std::endl;
        } else if (params.collisionType2 == MRT) {
            std::cout << "using mrt collision for ca equation" << std::endl;
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
        } else if (params.collisionType1 == MRT) {
            init1eq<EquationTag1, MRTCollider<dim, npop>>();
        }

        // init concentration distrib A
        if (params.collisionType2 == BGK) {
            init1eq<EquationTag2, BGKCollider<dim, npop>>();
        } else if (params.collisionType2 == MRT) {
            init1eq<EquationTag2, MRTCollider<dim, npop>>();
        }
    }

    void update_f() override
    {

        // update phi
        if (params.collisionType1 == BGK) {
            update1eq<EquationTag1, BGKCollider<dim, npop>>();
        } else if (params.collisionType1 == MRT) {
            update1eq<EquationTag1, MRTCollider<dim, npop>>();
        }

        // update ns
        if (params.collisionType2 == BGK) {
            update1eq<EquationTag2, BGKCollider<dim, npop>>();
        } else if (params.collisionType2 == MRT) {
            update1eq<EquationTag2, MRTCollider<dim, npop>>();
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
        umax = -scheme.Model.gy * scheme.Model.time;

        real_t possible_dt = scheme.Model.fMach * scheme.Model.dx / sqrt(3) / sqrt(umax * umax - scheme.Model.gy * scheme.Model.L);

        scheme.Model.dt = possible_dt;
    }

    void update_total_time(real_t m_t) override
    {
        scheme.Model.time = m_t;
    };

    void make_boundaries() override
    {

        BoundaryFunctor<dim, npop, LBMSchemeSpec>::make_all_boundaries(params, scheme);

        this->bcPerMgr.make_boundaries(scheme.f1, BOUNDARY_EQUATION_1);
        this->bcPerMgr.make_boundaries(scheme.f2, BOUNDARY_EQUATION_2);
        this->bcPerMgr.make_boundaries(scheme.f3, BOUNDARY_EQUATION_3);
    }

    real_t get_dt() { return scheme.Model.dt; }

}; // end class ProblemBase

}; //end namespace
#endif // LBM_SCHEME_BASE_H_
