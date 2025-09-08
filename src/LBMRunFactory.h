#ifndef LBM_RUN_FACTORY_H_
#define LBM_RUN_FACTORY_H_

#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include "LBMParams.h" // for enums
class ConfigMap;

#include "LBMRunBase.h"



template<int dim, int npop>
struct DQindex
{
	    //! retrieve number of space dimension (2 or 3)
    static constexpr int get_dim()
    {
        return dim;
    };
    //! retrieve number of population (aka number of distribution functions)
    static constexpr int get_npop()
    {
        return npop;
    };
};

using LBMRunTypes = std::tuple<
  DQindex<2, NPOP_D2Q5>,
  DQindex<2, NPOP_D2Q9>,
  DQindex<3, NPOP_D3Q7>,
  DQindex<3, NPOP_D3Q15>,
  DQindex<3, NPOP_D3Q19>,
  DQindex<3, NPOP_D3Q27>
>;

/**
 * An abstract base class to define a common interface for concrete LBMRuns.
 *
 * The main purpose is to return a concrete LBMRun object.
 * The idea here it to define a map between a name and the actual LBMRun.
 *
 * Each derived class will have to define from this class.
 *
 */
class LBMRunFactory {

private:
    // make constructor private -- this class is singleton
    LBMRunFactory();
    LBMRunFactory(const LBMRunFactory&) = delete; // non construction-copyable
    LBMRunFactory& operator=(const LBMRunFactory&) { return *this; } // non-copyable

    /**
	 * typedef to the LBMRun creation function pointer.
	 * This function pointer will actually be populated with a concrete LBMRun
	 * method named "create" which takes in input a LBMParams pointer
	 * (necessary to call the concrete LBMRun constructor).
	 */
    using LBMRunCreateFn = LBMRunBase* (*)(LBMParams& params,
        ConfigMap& configMap);

    /**
	 * Map to associate a label with a pair of LBMRun creation function, and
	 * UserDataManager creation function.
	 * Each concrete LBMRun / UserDataManger class must provide a (static)
	 * creation method named create.
	 */
    using LBMRunCreateMap = std::map<std::string, LBMRunCreateFn>;
    LBMRunCreateMap m_LBMRunCreateMap;

public:
    ~LBMRunFactory() { m_LBMRunCreateMap.clear(); }

    static LBMRunFactory& Instance()
    {
        static LBMRunFactory instance;
        return instance;
    }

    /**
	 * Routine to insert an LBMRun function into the map.
	 * Note that this register function can be used to serve
	 * at least two different purposes:
	 * - in the concrete factory: register existing callback's
	 * - in some client code, register a callback from a plugin code, at runtime.
	 */
    void registerLBMRun(const std::string& key, LBMRunCreateFn cfn)
    {
        m_LBMRunCreateMap[key] = cfn;
    };

    /**
	 * \brief Retrieve one of the possible LBMRuns by name.
	 *
	 * Allowed default names are defined in the concrete factory.
	 */
    LBMRunBase* create(const std::string& LBMRun_name,
        LBMParams& params,
        ConfigMap& configMap)
    {

        // find the LBMRun name in the register map
        LBMRunCreateMap::iterator it = m_LBMRunCreateMap.find(LBMRun_name);

        // if found, just create and return the LBMRun object
        if (it != m_LBMRunCreateMap.end()) {

            // create a LBMRun
            LBMRunBase* lbm = it->second(params, configMap);

            // additionnal initialization (each LBMRun might override this method)
            // LBMRun->init_io();

            return lbm;
        }

        // if not found, return null pointer
        // it is the responsability of the client code to deal with
        // the possibility to have a nullptr callback (does nothing).
        printf("############ WARNING: ############\n");
        printf("%s: is not recognized as a valid application name key.\n", LBMRun_name.c_str());
        printf("Valid LBMRun names are:\n");
        for (auto it = m_LBMRunCreateMap.begin(); it != m_LBMRunCreateMap.end(); ++it)
            printf("%s\n", it->first.c_str());
        printf("############ WARNING: ############\n");

        printf("LBMRun application name not found\n");
        std::abort();

        return nullptr;

    }; // create

}; // class LBMRunFactory

#endif // LBM_RUN_FACTORY_H_
