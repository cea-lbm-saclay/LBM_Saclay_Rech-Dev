#ifndef LBM_MAPS_H_
#define LBM_MAPS_H_

#include "LBM_enums.h"
#include <unordered_map>

using str2int_t = std::unordered_map<std::string, int>;
using int2str_t = std::unordered_map<int, std::string>;
struct maps
{

    static str2int_t getMAP_COLLISION_TYPES()
    {
        str2int_t MAP_COLLISION_TYPES = {
            { "BGK", BGK },
            { "TRT", TRT },
            { "MRT", MRT }
        };
        return MAP_COLLISION_TYPES;
    };

    static str2int_t getMAP_TRT_TAU_METHOD()
    {
        str2int_t MAP_TRT_TAU_METHOD = {
            { "fixed_tau", FIXED_TAU },
            { "conditional_tau", CONDITIONAL_TAU },
            { "fixed_lambda", FIXED_LAMBDA }
        };
        return MAP_TRT_TAU_METHOD;
    };

    static str2int_t getMAP_BC_NAMES_TO_ID()
    {
        str2int_t MAP_BC_NAMES_TO_ID = {
            { "periodic", BC_PERIODIC },
            { "zero_flux", BC_ZERO_FLUX },
            { "antibounceback", BC_ANTI_BOUNCE_BACK },
            { "bounceback", BC_BOUNCEBACK },
            { "poiseuille", BC_POISEUILLE },
            { "free_flow", BC_FREEFLOW }
        };
        return MAP_BC_NAMES_TO_ID;
    };

    static str2int_t getMAP_EQS_NAMES_TO_ID()
    {
        str2int_t MAP_EQS_NAMES_TO_ID = {
            { "equation1", BOUNDARY_EQUATION_1 },
            { "equation2", BOUNDARY_EQUATION_2 },
            { "equation3", BOUNDARY_EQUATION_3 },
            { "equation4", BOUNDARY_EQUATION_4 }
        };

        return MAP_EQS_NAMES_TO_ID;
    };

    static int2str_t getMAP_EQS_ID_TO_NAMES()
    {
        int2str_t MAP_EQS_ID_TO_NAMES = {
            { BOUNDARY_EQUATION_1, "equation1" },
            { BOUNDARY_EQUATION_2, "equation2" },
            { BOUNDARY_EQUATION_3, "equation3" },
            { BOUNDARY_EQUATION_4, "equation4" }
        };
        return MAP_EQS_ID_TO_NAMES;
    };

    static int2str_t getMAP_FACE_ID_TO_NAME()
    {
        int2str_t MAP_FACE_ID_TO_NAME = {
            { FACE_XMIN, "xmin" },
            { FACE_XMAX, "xmax" },
            { FACE_YMIN, "ymin" },
            { FACE_YMAX, "ymax" },
            { FACE_ZMIN, "zmin" },
            { FACE_ZMAX, "zmax" }
        };
        return MAP_FACE_ID_TO_NAME;
    };

    static str2int_t getMAP_FACE_NAME_TO_ID()
    {
        str2int_t MAP_FACE_NAME_TO_ID = {
            { "xmin", FACE_XMIN },
            { "xmax", FACE_XMAX },
            { "ymin", FACE_YMIN },
            { "ymax", FACE_YMAX },
            { "zmin", FACE_ZMIN },
            { "zmax", FACE_ZMAX }
        };

        return MAP_FACE_NAME_TO_ID;
    };
};

#endif // LBM_ENUMS_H_
