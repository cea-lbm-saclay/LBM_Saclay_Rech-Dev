#ifndef PROBLEM_GPMIXT_NS_H_
#define PROBLEM_GPMIXT_NS_H_

#include <Kokkos_Random.hpp>
#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>

#endif // __CUDA_ARCH__

#include "FieldManager.h"
#include "LBMParams.h"
#include "LBMScheme_GPMixt_NS.h"
#include "LBM_Base_Functor.h"
namespace PBM_GP_MIXT_NS {
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
    using Kernel2 = SchemeFunctor<dim, npop, LBMSchemeSpec, EquationTag2, BGKCollider<npop>>;
    using Kernel3 = SchemeFunctor<dim, npop, LBMSchemeSpec, EquationTag3, BGKCollider<npop>>;

    LBMParams params;

    LBMSchemeSpec scheme;

    Problem(ConfigMap& configMap, LBMParams params)
        : ProblemBase<dim, npop>(configMap, params, index2names::get_id2names())
        , params(params)
    {
        Base::nbVar = COMPONENT_SIZE;
        Base::nbEqs = 3;

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

    void init_m() override
    {

        if (params.init_str == "data") {
            loadDataCSV(scheme.lbm_data);
        }

        // init Macro
        MacroInitKernel::template apply<TagInitMacro>(params, scheme);
        MacroKernel::template apply<TagUpdateMacroGrads>(params, scheme);
    }
    void init_f() override
    {

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

        // update concentration distrib
        Kernel2::template apply<TagUpdate>(params, scheme);
        EquationTag2 tag2;
        scheme.swap_distribution(tag2);

        // update fluid distrib
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

        BoundaryFunctor<dim, npop, LBMSchemeSpec>::make_all_boundaries(params, scheme);

        this->bcPerMgr.make_boundaries(scheme.f1, BOUNDARY_EQUATION_1);
        this->bcPerMgr.make_boundaries(scheme.f2, BOUNDARY_EQUATION_2);
        this->bcPerMgr.make_boundaries(scheme.f3, BOUNDARY_EQUATION_3);
    }

    void update_dt() override
    {
    }

    void update_total_time(real_t m_t) override
    {
        scheme.Model.time = m_t;
    };

    real_t get_dt() { return scheme.Model.dt; }

    void loadDataCSV(LBMArray2d ldata) override
    {
        const int gw = params.ghostWidth;

        std::string filename = params.initFileName;
        std::ifstream myFile(filename);

        int sizeRatio = params.sizeRatio;
        printf("sizeRatio  :  %d\n", sizeRatio);

        // temporary host array to read data before copying into device array
        const int nx = params.nx;
        const int ny = params.ny;
        ArrayScalar2d readData_device = ArrayScalar2d("readData", nx, ny);
        ArrayScalar2dHost readData = Kokkos::create_mirror_view(readData_device);

        // Helper vars
        std::string line;
        int val;
        Kokkos::Array<int, 4> indval;
        // Read data, line by line
        while (std::getline(myFile, line)) {
            // Create a stringstream of the current line
            std::stringstream ss(line);

            // Keep track of the current column index
            int colIdx = 0;

            // Extract each integer
            while (ss >> val) {
                // Add the current integer to the 'colIdx' column's values vector
                indval[colIdx] = val;
                //printf("%d  :  %d\n" ,colIdx, val);
                // If the next token is a comma, ignore it and move on
                if (ss.peek() == ',')
                    ss.ignore();
                // Increment the column index
                colIdx++;
            }
            //printf("%d  ,  %d,  %d,  %d\n" ,indval[0],indval[1],indval[2],indval[3]);
            if (dim == TWO_D) {
                for (int i = 0; i < sizeRatio; i++)
                    for (int j = 0; j < sizeRatio; j++)
                        readData(sizeRatio * indval[0] + i, sizeRatio * indval[1] + j) = indval[3];
            }
        }

        // Close file
        myFile.close();

        // take a subview in LBM data where we want to copy read data into
        auto ldata_phi = Kokkos::subview(ldata, std::make_pair(gw, gw + nx), std::make_pair(gw, gw + ny), int(IPHI));

        // copy read data into LBM device data
        Kokkos::deep_copy(readData_device, readData);
        Kokkos::deep_copy(ldata_phi, readData_device);

    } // loadDataCSV - 2d;
    void loadDataCSV(LBMArray3d ldata) override {};

}; // end class ProblemBase

} //end namespace
#endif // LBM_SCHEME_BASE_H_
