#ifndef INDEX_GPMIXT_TERNARY_H_
#define INDEX_GPMIXT_TERNARY_H_

#include "LBM_maps.h"

namespace PBM_GP_MIXT_TERNARY {

enum ComponentIndex {
    ID, /*!< ID Density / Concentration field index */
    IU, /*!< X velocity / momentum index */
    IV, /*!< Y velocity / momentum index */
    IW, /*!< Z velocity / momentum index */
    IPHI, /*!< Phase field index */
    ICC, // for connected components
    IDPHIDT,
    IP, /*!< Pressure -- used when Navier-Stokes enabled */
    IC,
    ICB,
    IMU,
    IMUB,
    IDPHIDX,
    IDPHIDY,
    IDPHIDZ,
    ILAPLAPHI,
    IDMUDT,
    ISUPERSAT,
    IGRANDPOTENTIAL,
    I_FREE_ENERGY,
    IINTERFACIALENERGY,
    IDELTAW,
    IDCADX,
    IDCADY,
    IDCADZ,
    IDCBDX,
    IDCBDY,
    IDCBDZ,
    IDUMMY,
    COMPONENT_SIZE /*!< invalid index, just counting number of fields */
};

struct index2names {
    static int2str_t get_id2names()
    {

        int2str_t map;

        // insert some fields
        map[ID] = "rho";
        map[IU] = "vx";
        map[IV] = "vy";
        map[IW] = "vz";
        map[IPHI] = "phi";
        map[ICC] = "connected_components";
        map[IP] = "pressure";
        map[IC] = "composition";
        map[IMU] = "mu";
        map[ICB] = "cB";
        map[IMUB] = "muB";
        map[IDPHIDX] = "dphidx";
        map[IDPHIDY] = "dphidy";
        map[IDPHIDZ] = "dphidz";
        map[ILAPLAPHI] = "laplaphi";
        map[IDPHIDT] = "dphidt";
        map[IDMUDT] = "dmudt";
        map[ISUPERSAT] = "supersat";
        map[IGRANDPOTENTIAL] = "grand_potential";
        map[I_FREE_ENERGY] = "free_energy";
        map[IINTERFACIALENERGY] = "interfacial_energy";
        map[IDELTAW] = "dw";
        map[IDCADX] = "dcadx";
        map[IDCADY] = "dcady";
        map[IDCADZ] = "dcadz";
        map[IDCBDX] = "dcbdx";
        map[IDCBDY] = "dcbdy";
        map[IDCBDZ] = "dcbdz";
        return map;

    } // get_id2names_all
};
}; //end namespace
#endif // INDEX_NS_PF_H_
