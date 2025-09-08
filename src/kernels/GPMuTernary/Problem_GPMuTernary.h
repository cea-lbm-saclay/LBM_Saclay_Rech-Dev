#ifndef PROBLEM_GPMU_TERNARY_H_
#define PROBLEM_GPMU_TERNARY_H_

#include <Kokkos_Random.hpp>
#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>

#endif // __CUDA_ARCH__

#include "FieldManager.h"
#include "LBMParams.h"
#include "LBMScheme_GPMu_Ternary.h"
#include "LBM_Base_Functor.h"

namespace PBM_GP_MU_TERNARY {
template <int dim, int npop, typename modelType>
struct Problem : public ProblemBase<dim, npop> {
    using Base = ProblemBase<dim, npop>;

    using id2index_t = Kokkos::Array<int, COMPONENT_SIZE>;
    using LBMArray = typename LBMBaseFunctor<dim, npop>::LBMArray;
    using LBMArrayHost = typename LBMArray::HostMirror;
    using FArray = typename LBMBaseFunctor<dim, npop>::FArray;
    using FState = typename Kokkos::Array<real_t, npop>;
    static constexpr real_t NORMAL_EPSILON = 1.0e-15;

    using LBMSchemeSpec = LBMScheme<dim, npop, modelType>;
    using MacroKernel = MacroFunctor<dim, npop, LBMSchemeSpec>;
    using MacroInitKernel = MacroInitFunctor<dim, npop, LBMSchemeSpec>;
    using Kernel1 = SchemeFunctor<dim, npop, LBMSchemeSpec, EquationTag1, BGKCollider<npop>>;
    using Kernel2 = SchemeFunctor<dim, npop, LBMSchemeSpec, EquationTag2, BGKColliderTimeFactor<npop>>;
    using Kernel3 = SchemeFunctor<dim, npop, LBMSchemeSpec, EquationTag3, BGKColliderTimeFactor<npop>>;

    LBMParams params;

    LBMSchemeSpec scheme;

    Problem(ConfigMap& configMap, LBMParams params)
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

    void init() override
    {

        //~ std::string init_str = std::string(configMap.getString("init", "init_type", "unknown"));
        //~ if (init_str == "data") {loadImageData(data);}

        // init Macro
        MacroInitKernel::template apply<TagInitMacro>(params, scheme);
        MacroKernel::template apply<TagUpdateMacroGrads>(params, scheme);

        // init distribution functions
        Kernel1::template apply<TagInit>(params, scheme);
        Kernel2::template apply<TagInit>(params, scheme);
        Kernel3::template apply<TagInit>(params, scheme);
    }

    void update_f() override
    {

        // update phase field distrib
        Kernel1::template apply<TagUpdate>(params, scheme);
        EquationTag1 tag1;
        scheme.swap_distribution(tag1);

        // update concentration a distrib
        Kernel2::template apply<TagUpdate>(params, scheme);
        EquationTag2 tag2;
        scheme.swap_distribution(tag2);

        // update concentration b distrib
        Kernel3::template apply<TagUpdate>(params, scheme);
        EquationTag3 tag3;
        scheme.swap_distribution(tag3);
    }

    void update_m() override
    {

        // update macro fields
        MacroKernel::template apply<TagUpdateMacro>(params, scheme);
        MacroKernel::template apply<TagUpdateMacroGrads>(params, scheme);
    }

    void make_boundaries() override
    {

        BoundaryFunctor<dim, npop, LBMSchemeSpec>::apply(params, scheme, FACE_XMIN);
        BoundaryFunctor<dim, npop, LBMSchemeSpec>::apply(params, scheme, FACE_XMAX);
        BoundaryFunctor<dim, npop, LBMSchemeSpec>::apply(params, scheme, FACE_YMIN);
        BoundaryFunctor<dim, npop, LBMSchemeSpec>::apply(params, scheme, FACE_YMAX);
        if (dim == 3) {
            BoundaryFunctor<dim, npop, LBMSchemeSpec>::apply(params, scheme, FACE_ZMIN);
            BoundaryFunctor<dim, npop, LBMSchemeSpec>::apply(params, scheme, FACE_ZMAX);
        }

        this->bcPerMgr.make_boundaries(scheme.f1, BOUNDARY_EQUATION_1);
        this->bcPerMgr.make_boundaries(scheme.f2, BOUNDARY_EQUATION_2);
        this->bcPerMgr.make_boundaries(scheme.f3, BOUNDARY_EQUATION_3);
    }

}; // end class ProblemBase

}

#endif // LBM_SCHEME_BASE_H_
