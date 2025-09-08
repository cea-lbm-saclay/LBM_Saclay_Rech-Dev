#ifndef INDEX_AD_H_
#define INDEX_AD_H_

namespace PBM_AD {

enum ComponentIndex {
    IU, /*!< X velocity / momentum index */
    IV, /*!< Y velocity / momentum index */
    IW, /*!< Z velocity / momentum index */
    IC, /*!< Phase field index */
    IT, /*!< Time index */
    COMPONENT_SIZE /*!< invalid index, just counting number of fields */
};

struct index2names {
    static int2str_t get_id2names()
    {

        int2str_t map;

        // insert some fields
        map[IC] = "comp";
        map[IU] = "vx";
        map[IV] = "vy";
        map[IW] = "vz";
        map[IT] = "t";

        return map;

    }; // get_id2names_all
};
}; //end namespace
#endif // INDEX_NS_PF_H_
