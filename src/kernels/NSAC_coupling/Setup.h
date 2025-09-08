/**
 * \file Setup.h
 * \brief Register the problem so that it may be created from LBMRun.
 * 
 * How to write this setup header ?
 * 
 * Includes :
 * This header must include "../ProblemFactory.h" which gives the 
 * tool to register the problem
 * An implementation of a class "Problem" must also be included, which
 * has to be derived from ProblemBase and must be templated on 
 * at least 2 int (dim and npop)
 * 
 * Step 1:
 * First define the pbm_registerer struct for the current problem.
 * It needs to define a constructor with no arguments.
 * In this constructor, we can register the problems we created using 
 * the register_problem function from ProblemFactory.h
 * To do this, we must give either a template of a problem class that is templated on <int dim, int npop> 
 *      in this case we just do register_problem<Problem>( pbm_name, submodel_name)
 * or a more complex template <int dim, int npop, typename... t_params> where t_params can be multiple template parameters
 * and in this case we must give the additional t_params to the register_problem function:
 *      in this case, we do register_problem<Problem, tparams...>( pbm_name, submodel_name)
 * 
 * 
 * Step 2:
 * We simply create a static instance of the pbm_registerer.
 * It will be default-initialized with the default constructor, which is the one we defined earlier, when LBM_saclay is executed.
 * The actual registering takes place at this time (startup of executable)
 */
#ifndef SETUP_NSAC_COUPLING_H_   // change this
#define SETUP_NSAC_COUPLING_H_   // change this

#include "../ProblemFactory.h"
#include "ProblemNSAC.h"        // change this line

namespace PBM_NSAC {  // change the namespace name
// this is step 1
struct pbm_registerer
{
    
    
    pbm_registerer()
    {

        register_problem<Problem, ModelParams::Tag2Quadra>("NSAC_coupling", "base"); // change this
    };

};

// this is step 2
static pbm_registerer registerer;

}
#endif //SETUP_NSAC_COUPLING_H_
