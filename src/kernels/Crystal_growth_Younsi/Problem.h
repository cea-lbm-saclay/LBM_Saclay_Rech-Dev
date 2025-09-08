#ifndef PROBLEM_GPMIXT_H_
#define PROBLEM_GPMIXT_H_

#include <Kokkos_Random.hpp>
#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>

#endif // __CUDA_ARCH__

#include "FieldManager.h"
#include "LBMParams.h"
#include "LBMScheme.h"
#include "LBM_Base_Functor.h"
#include "../../io/io_csv.h"
namespace PBM_DIRECTIONAL_SOLIDIFICATION {
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
        MacroKernel::template apply<TagUpdateMacroGrads>(params, scheme);
    }

    void init_f() override
    {

        // init distribution functions
        if (params.collisionType1 == BGK) {
            init1eq<EquationTag1, BGKColliderTimeFactor<dim, npop>>();
        } 

        if (params.collisionType2 == BGK) {
            init1eq<EquationTag2, BGKCollider<dim, npop>>();
        } 
    }

    void update_f() override
    {
        // update phase field distrib
        if (params.collisionType1 == BGK) {
            update1eq<EquationTag1, BGKColliderTimeFactor<dim, npop>>();
        } 

        // update concentration distrib
        if (params.collisionType2 == BGK) {
            update1eq<EquationTag2, BGKCollider<dim, npop>>();
        } 
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

        Base::bcPerMgr.make_boundaries(scheme.f1, BOUNDARY_EQUATION_1);
        Base::bcPerMgr.make_boundaries(scheme.f2, BOUNDARY_EQUATION_2);
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



}; // end class ProblemBase

} // end namespace

#endif // LBM_SCHEME_BASE_H_
