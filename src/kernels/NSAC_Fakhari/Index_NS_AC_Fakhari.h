#ifndef INDEX_NS_PF_F_H_
#define INDEX_NS_PF_F_H_

namespace PBM_NS_AC_Fakhari {

enum ComponentIndex {
    ID, /*!< ID Density / Concentration field index */
    IP,
    IU, /*!< X velocity / momentum index */
    IV, /*!< Y velocity / momentum index */
    IPHI, /*!< Phase field index */
    ILAPLAPHI,
    IDPHIDX,
    IDPHIDY,
    IDPHIDZ,
    IW, /*!< Z velocity / momentum index */
    IFX, /*!< Force Navier Stokes X */
    IFY, /*!< Force Navier Stokes Y */
    IT, /*!< Time index */
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
        map[IFX] = "forceNSX";
        map[IFY] = "forceNSY";

        map[IDPHIDX] = "dphidx";
        map[IDPHIDY] = "dphidy";
        map[IDPHIDZ] = "dphidz";
        map[ILAPLAPHI] = "laplaphi";
        map[IT] = "t";
        return map;

    }; // get_id2names_all
};
}; //end namespace
#endif // INDEX_NS_PF_F_H_
