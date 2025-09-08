#ifndef INDEX_NS_2AC_C_H_
#define INDEX_NS_2AC_C_H_
namespace PB_NS{
enum ComponentIndex {
    ID, /*!< ID Density index */
    IP, /*!< Pressure   index */
    IU, /*!< X velocity / momentum index */
    IV, /*!< Y velocity / momentum index */
    IPHI, /*!< First Phase field index */
    IPHI2,
    /*!< Second Phase field index */ // AJOUT ALAIN
    IPHI3,
    /*!< Third Phase field index */ // AJOUT ALAIN
    IC, /*!< Composition field index */
    IMU, /*!< Chemical index */
    IDCDX, /*!< Grad X of Composition field */
    IDCDY, /*!< Grad Y of Composition field */
    IDPHIDX, /*!< Grad X of first Phase field */
    IDPHIDY, /*!< Grad Y of first Phase field */
    IDPHI2DX,
    /*!< Grad X of second Phase field */ // AJOUT ALAIN
    IDPHI2DY,
    /*!< Grad Y of second Phase field */ // AJOUT ALAIN
    IDPHI3DX,
    /*!< Grad X of third Phase field */ // AJOUT ALAIN
    IDPHI3DY,
    /*!< Grad Y of third Phase field */ // AJOUT ALAIN
    IDRHODX,
    /*!< Grad X of density */ // AJOUT ALAIN
    IDRHODY,
    /*!< Grad Y of density */ // AJOUT ALAIN
    ILAPLAPHI, /*!< Laplacien of Phase field */
    ILAPLAPHI2,
    /*!< Laplacien of Phase field */ // AJOUT ALAIN
    ILAPLAPHI3,
    /*!< Laplacien of Phase field */ // AJOUT ALAIN
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
        map[IPHI] = "phi";
        map[IPHI2] = "phi2";
        map[IPHI3] = "phi3"; // AJOUT ALAIN
        map[IP] = "pressure";
        map[IC] = "composition";
        map[IMU] = "potential";
        map[IFX] = "forceNSX";
        map[IFY] = "forceNSY";
        map[IT] = "t";

        map[IDCDX] = "dcdx";
        map[IDCDY] = "dcdy";
        map[IDPHIDX] = "dphidx";
        map[IDPHIDY] = "dphidy";
        map[IDPHI2DX] = "dphi2dx";
        map[IDPHI2DY] = "dphi2dy";
        map[IDPHI3DX] = "dphi3dx";
        map[IDPHI3DY] = "dphi3dy";
        map[ILAPLAPHI] = "laplaphi";
        return map;

    }; // get_id2names_all
};

//~ enum InitTypes {
//~ DUMMY,
//~ THREEPHASES_SPLASHING_DROPLETS,
//~ THREEPHASES_RISING_DROPLETS,
//~ THREEPHASES_CONTAINER,
//~ THREEPHASES_RAYLEIGH_TAYLOR,
//~ THREEPHASES_SPINODAL_DECOMPOSITION,
//~ THREEPHASES_SPREADING_LENS,
//~ THREEPHASES_CAPSULE,
//~ };
}
#endif
