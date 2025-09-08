#ifndef PROBLEM_FACTORY_H_
#define PROBLEM_FACTORY_H_

#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <tuple>

#include <type_traits>
#include <utility>

#include "LBMRunFactory.h"
#include "ProblemBase.h"

template<int dim, int npop>
class ProblemFactory
{

private:
    // make constructor private -- this class is singleton
    ProblemFactory() {};
    ProblemFactory(const ProblemFactory&) = delete; // non construction-copyable
    ProblemFactory& operator=(const ProblemFactory&) { return *this; } // non-copyable

    /**
     * typedef of a Problem creation function pointer.
     */
    using ProblemCreateFn = ProblemBase<dim, npop>* (*)(ConfigMap& configMap, LBMParams& params);

    /**
     * 2 dimensional map to associate a problem name and submodel name with a creation function
     */
    using ModelMap = std::map<std::string, ProblemCreateFn>;
    using ProblemCreateMap = std::map<std::string, ModelMap>;

    ProblemCreateMap m_ProblemCreateMap;

public:
    ~ProblemFactory() { m_ProblemCreateMap.clear(); }

    static ProblemFactory<dim, npop>& Instance()
    {
        static ProblemFactory<dim, npop> instance;
        return instance;
    };

    /**
     * Routine to insert a problem creator function into the map.
     * Note that this register function can be used to serve
     * at least two different purposes:
     * - in the concrete factory: register existing callback's
     * - in some client code, register a callback from a plugin code, at runtime.
     */
    bool register_pbm(const std::string& keyPb,const std::string& keyModel, const ProblemCreateFn cfn)
    {
        // check for existence of the problem, to detect overwriting due to wrong naming of new problems
        typename ProblemCreateMap::iterator it = m_ProblemCreateMap.find(keyPb);
        if (it != m_ProblemCreateMap.end())
        {
            typename ModelMap::iterator itModel = it->second.find(keyModel);
            if (itModel != it->second.end())
            {
                if (cfn != itModel->second)
                {
                    printf("############ WARNING: ############\n");
                    printf("overwriting of problem/submodel register\n");
                    printf("for problem %s, submodel %s\n", keyPb.c_str(), keyModel.c_str());

                    return false;
                }
            }
        }
        m_ProblemCreateMap[keyPb][keyModel] = cfn;
        return true;
    };

    /**
     * \brief Retrieve one of the possible problems by name.
     *
     * Allowed default names are defined in the concrete factory.
     */
    ProblemBase<dim, npop>* create(const std::string& keyPb,const std::string& keyModel,
                                   LBMParams& params,
                                   ConfigMap& configMap)
    {

        // find the Problem name in the register map
        typename ProblemCreateMap::iterator it = m_ProblemCreateMap.find(keyPb);

        // if found, find the model name in the subproblem map
        if (it != m_ProblemCreateMap.end())
        {

            typename ModelMap::iterator itModel = it->second.find(keyModel);

            // if found, create and return the problem
            if (itModel != it->second.end())
            {
                // create a problem
                ProblemBase<dim, npop>* pbm = itModel->second(configMap, params);

                // additionnal initialization (each LBMRun might override this method)
                // LBMRun->init_io();

                return pbm;
            }
        }

        // if not found, return null pointer
        // it is the responsability of the client code to deal with
        // the possibility to have a nullptr callback (does nothing).
        printf("############ WARNING: ############\n");
        printf("%s: is not recognized as a valid problem name key.\n", keyPb.c_str());
        enumerate_options();
        printf("############ WARNING: ############\n");

        printf("problem and/or model type name not found\n");
        std::abort();

        return nullptr;

    }; // create



    void enumerate_options() const
    {
        printf("##########################\n");
        printf("Valid problem names and submodels for this compilation are:\n");
        for (auto it= m_ProblemCreateMap.begin(); it != m_ProblemCreateMap.end(); ++it)
        {
            printf("%s\n", it->first.c_str());

            for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2)
            {
                printf("  - %s\n", it2->first.c_str());
            }
        }
        printf("##########################\n");

    }

}; // class ProblemFactory


// next two function are used to loop over a list of types, generated using std::tuple<...list>. They should be moved to another file, and they could be used in LBMRunFactory to register all the correct LBMRun classes.
template<std::size_t... indices, class LoopBody>
void loop_impl(std::index_sequence<indices...>, LoopBody&& loop_body)
{
    (// C++17's fold expression
        loop_body(std::integral_constant<std::size_t, indices> {}),
        ...
    );
}

template<std::size_t N, class LoopBody>
void loop(std::integral_constant<std::size_t, N>, LoopBody&& loop_body)
{
    loop_impl(std::make_index_sequence<N> {}, std::forward<LoopBody>(loop_body));
}



// template for problem creation function
template<typename Base, typename Derived>
Base* create(ConfigMap& configMap, LBMParams& params)
{
    Base* pbm =new Derived(configMap, params);
    return pbm;
}


// function used to register problems from outside. It automatically loops over all defined LBM lattice, see LBMRunFactory.h
// with the actual usage in Setup headers, problems are registered twice (and maybe more in the future) because this is called at every include of the Setup header from a cpp file. Only way to avoid that is by using a Setup.cpp in each problem directory, which I want to avoid for user simplicity.
template<template<int, int, typename... mT> typename problem_with_model, typename... modelType>
void register_problem(const std::string& pbm_name, const std::string& model_name)
{
    loop(std::tuple_size<LBMRunTypes> {}, [&] (auto i)
    {
        using T = std::tuple_element_t<i, LBMRunTypes>;

        using pbm_base = ProblemBase<T::get_dim(),T::get_npop()>;
        using pbm_type = problem_with_model<T::get_dim(),T::get_npop(), modelType...>;

        bool success =ProblemFactory<T::get_dim(),T::get_npop()>::Instance().register_pbm(pbm_name, model_name, &create<pbm_base,pbm_type>);

        if (!success)
        {
            printf("warning when registering problem\n");
            printf("problem class signature is %s\n",typeid(pbm_type).name());
            std::abort();
        }
        else
        {
            //printf("successfully registered problem\n");
            //printf("problem class signature is %s\n",typeid(pbm_type).name());
        }

    });


};

#endif // PROBLEM_FACTORY_H_
