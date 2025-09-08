#ifndef INDEX_AC_H_
#define INDEX_AC_H_

namespace PBM_AC {

enum ComponentIndex {
    IU, /*!< X velocity / momentum index */
    IV, /*!< Y velocity / momentum index */
    IW, /*!< Z velocity / momentum index */
    IPHI, /*!< Phase field index */
    ILAPLAPHI,
    IDPHIDX,
    IDPHIDY,
    IDPHIDZ,
    COMPONENT_SIZE /*!< invalid index, just counting number of fields */
};

struct index2names {
    static int2str_t get_id2names()
    {

        int2str_t map;

        // insert some fields
        map[IU] = "vx";
        map[IV] = "vy";
        map[IW] = "vz";
        map[IPHI] = "phi";
        map[IDPHIDX] = "dphidx";
        map[IDPHIDY] = "dphidy";
        map[IDPHIDZ] = "dphidz";
        map[ILAPLAPHI] = "laplaphi";
        return map;

    }; // get_id2names_all
};
}; //end namespace
#endif // INDEX_NS_PF_H_
