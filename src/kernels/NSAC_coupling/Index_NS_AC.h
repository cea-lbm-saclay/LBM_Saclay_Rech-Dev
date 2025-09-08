/**
 * \file Index_NS_AC.h
 * \brief Define macro variable names for NSAC problem
 *
 */
#ifndef INDEX_NS_PF_H_
#define INDEX_NS_PF_H_

namespace PBM_NSAC {

enum ComponentIndex {
    ID, /*!< ID Density / Concentration field index */
    IP,
    IU, /*!< X velocity / momentum index */
    IV, /*!< Y velocity / momentum index */
    IW, /*!< Z velocity / momentum index */
    IPHI, /*!< Phase field index */
    ILAPLAPHI,
    IDPHIDX,
    IDPHIDY,
    IDPHIDZ,
    ICC,

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
        map[IP] = "pressure";

        map[IDPHIDX] = "dphidx";
        map[IDPHIDY] = "dphidy";
        map[IDPHIDZ] = "dphidz";
        map[ILAPLAPHI] = "laplaphi";
        map[ICC] = "connected_components";
        return map;

    }; // get_id2names_all
};
}; //end namespace
#endif // INDEX_NS_PF_H_
