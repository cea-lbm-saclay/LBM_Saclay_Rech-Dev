#ifndef INDEX_NS_AC_C_H_
#define INDEX_NS_AC_C_H_

namespace PBM_NS_AC_Compo {

enum ComponentIndex {
  ID,            /*!< ID Density index */
  IP,            /*!< Pressure   index */
  IU,            /*!< X velocity / momentum index */
  IV,            /*!< Y velocity / momentum index */
  IW,            /*!< Z velocity / momentum index */
  IPHI,          /*!< Phase field index */
  IPHI2,         /*!< Phase field index for gas phase */
  IPSI,          /*!< Phase field index for solid phase */
  ISPHI,         /*!< Index for source term in phase phi */
  IC,            /*!< Composition field index */
  IMU,           /*!< Chemical index */
  IDCDX,         /*!< Grad X of Composition field */
  IDCDY,         /*!< Grad Y of Composition field */
  IDCDZ,         /*!< Grad Z of Composition field */
  IDPHIDX,       /*!< Grad X of Phase field */
  IDPHIDY,       /*!< Grad Y of Phase field */
  IDPHIDZ,       /*!< Grad Z of Phase field */
  IDPHI2DX,
  IDPHI2DY,
  IDPHI2DZ,
  IDPSIDX,
  IDPSIDY,
  IDPSIDZ,
  IDRHODX,       /*!< Grad X of density */
  IDRHODY,       /*!< Grad Y of density */
  IDRHODZ,       /*!< Grad Z of density */
  ILAPLAPHI,     /*!< Laplacien of Phase field */
  ILAPLAPHI2,
  ILAPLAPSI,
  IUB_X,
  IUB_Y,
  IUB_Z,
  IFX,           /*!< Force Navier Stokes X */
  IFY,           /*!< Force Navier Stokes Y */
  IFZ,           /*!< Force Navier Stokes Z */
  IT,            /*!< Time index */
  COMPONENT_SIZE /*!< invalid index, just counting number of fields */
};

struct index2names {
  static int2str_t get_id2names() {

    int2str_t map;

    // insert some fields
    map[ID]    = "rho";
    map[IU]    = "vx";
    map[IV]    = "vy";
    map[IW]    = "vz";
    map[IPHI]  = "phi";
    map[IPHI2] = "phi2";
    map[IPSI]  = "psi";
    map[ISPHI] = "source_phi";
    map[IP]    = "pressure";
    map[IC]    = "composition";
    map[IMU]   = "potential";
    map[IFX]   = "forceNSX";
    map[IFY]   = "forceNSY";
    map[IFZ]   = "forceNSZ";
    map[IUB_X] = "us_x";
    map[IUB_Y] = "us_y";
    map[IT]    = "t";

    map[IDCDX] = "dcdx";
    map[IDCDY] = "dcdy";
    map[IDCDZ] = "dcdz";
    map[IDPHIDX] = "dphidx";
    map[IDPHIDY] = "dphidy";
    map[IDPHIDZ] = "dphidz";
    map[ILAPLAPHI] = "laplaphi";
    // map[IDRHODX] = "drhodx";
    // map[IDRHODY] = "drhody";
    // map[IDRHODZ] = "drhodz";
    return map;

  }; // get_id2names_all
};
}; // namespace PBM_NS_AC_Compo
#endif
