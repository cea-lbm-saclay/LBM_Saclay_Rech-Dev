#ifndef INDEX_GPMIXT_H_
#define INDEX_GPMIXT_H_

namespace PBM_DIRECTIONAL_SOLIDIFICATION {

enum ComponentIndex {

    IPHI, /*!< Phase field index */
    IDPHIDT,
    ISUPERSAT,
    ICHI,
    IDPHIDX,
    IDPHIDY,
    IDPHIDZ,
    ILAPLAPHI,
    COMPONENT_SIZE /*! counting number of fields */
};

struct index2names {
    static int2str_t get_id2names()
    {

        int2str_t map;

        // insert some fields

        map[IPHI] = "phi";

        map[ISUPERSAT] = "supersat";
        map[ICHI] = "chi";

        map[IDPHIDX] = "dphidx";
        map[IDPHIDY] = "dphidy";
        map[IDPHIDZ] = "dphidz";
        map[ILAPLAPHI] = "laplaphi";
        map[IDPHIDT] = "dphidt";


        return map;

    } // get_id2names_all
};
}; //end namespace
#endif // INDEX_NS_PF_H_
