#ifndef INDEX_GPMIXT_H_
#define INDEX_GPMIXT_H_

namespace PBM_GP_MIXT {

enum ComponentIndex {

    IPHI, /*!< Phase field index */
    IDPHIDT,
    IC,
    IMU,
    IDPHIDX,
    IDPHIDY,
    IDPHIDZ,
    ILAPLAPHI,
    IDMUDT,
    IGRANDPOTENTIAL,
    IHELMOLTZENERGY,
    ICC,
    COMPONENT_SIZE /*! counting number of fields */
};

struct index2names {
    static int2str_t get_id2names()
    {

        int2str_t map;

        // insert some fields

        map[IPHI] = "phi";

        map[IC] = "composition";
        map[IMU] = "mu";

        map[IDPHIDX] = "dphidx";
        map[IDPHIDY] = "dphidy";
        map[IDPHIDZ] = "dphidz";
        map[ILAPLAPHI] = "laplaphi";
        map[IDPHIDT] = "dphidt";
        map[IDMUDT] = "dmudt";

        map[IGRANDPOTENTIAL] = "grand_potential";
        map[IHELMOLTZENERGY] = "helmoltz_energy";
        map[ICC] = "connected_components";

        return map;

    } // get_id2names_all
};
}; //end namespace
#endif // INDEX_NS_PF_H_
