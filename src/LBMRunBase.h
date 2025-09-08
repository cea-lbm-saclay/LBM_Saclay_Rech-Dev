/**
 * \file LBMRunBase.h
 */
#ifndef LBM_RUN_BASE_H_
#define LBM_RUN_BASE_H_

#include "kokkos_shared.h"

/**
 * Base class for LBMRun factory.
 */
class LBMRunBase {

public:
    LBMRunBase() = default;
    virtual ~LBMRunBase() = default;

    virtual void run() {};

    virtual real_t get_time() { return 0.0; };

}; // class LBMRunBase

#endif //LBM_RUN_BASE_H_
