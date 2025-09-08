#ifndef INDEX_GPMIXT_NS_H_
#define INDEX_GPMIXT_NS_H_

namespace PBM_GP_MIXT_NS {

enum ComponentIndex {
    ID, /*!< ID Density / Concentration field index */
    IU, /*!< X velocity / momentum index */
    IV, /*!< Y velocity / momentum index */
    IW, /*!< Z velocity / momentum index */
    IPHI, /*!< Phase field index */
    IDPHIDT,
    IP, /*!< Pressure -- used when Navier-Stokes enabled */
    IC,
    IMU,
    IDPHIDX,
    IDPHIDY,
    IDPHIDZ,
    ILAPLAPHI,
    IDMUDT,
    IGRANDPOTENTIAL,
    IHELMOLTZENERGY,
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

        return map;

    } // get_id2names_all
};
}; //end namespace
#endif // INDEX_NS_PF_H_
