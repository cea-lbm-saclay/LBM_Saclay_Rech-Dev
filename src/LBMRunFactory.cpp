#include "LBMRunFactory.h"

#include "LBMRun.h"


// The main solver creation routine
LBMRunFactory::LBMRunFactory()
{

    /*
	 * Register some possible LBMRun/UserDataManager.
	 */
    registerLBMRun("D2Q5", &LBMRun<2, NPOP_D2Q5>::create);
    registerLBMRun("D2Q9", &LBMRun<2, NPOP_D2Q9>::create);

    registerLBMRun("D3Q7", &LBMRun<3, NPOP_D3Q7>::create);
    registerLBMRun("D3Q15", &LBMRun<3, NPOP_D3Q15>::create);
    registerLBMRun("D3Q19", &LBMRun<3, NPOP_D3Q19>::create);
    registerLBMRun("D3Q27", &LBMRun<3, NPOP_D3Q27>::create);

} // LBMRunFactory::LBMRunFactory


