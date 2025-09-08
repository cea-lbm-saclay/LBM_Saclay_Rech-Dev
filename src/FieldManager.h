/**
 * \file FieldManager.h
 * \brief Define class FieldManager
 *
 * \date September, 3 2018
 */
#ifndef FIELD_MANAGER_H_
#define FIELD_MANAGER_H_

#include <unordered_map>

#include "LBMParams.h"
#include "LBM_enums.h"
#include "utils/config/ConfigMap.h"
//! a convenience alias to map variable names to id
using str2int_t = std::unordered_map<std::string, int>;

//! a convenience alias to map id to variable names
using int2str_t = std::unordered_map<int, std::string>;

/**
 * a convenience alias to map id (enum) to index used in LBMArray
 *
 * To clarify again:
 * - id is an enum value
 * - index is an integer between 0 and scalarFieldNb-1,
 *         used to access an LBMArray
 */
//~ using id2index_t = Kokkos::Array<int, COMPONENT_SIZE>;

/**
 * Field manager class.
 *
 * Initialize a std::unordered_map object to map enum ComponentIndex to actual integer
 * depending on runtime configuration (e.g. is Phase field activated, etc...).
 */
//~ class FieldManager {

//~ public:
//~ FieldManager() {};
//~ ~FieldManager() {};

//~ //! number of scalar field, will be used in allocating LBMArray
//~ int numScalarField = 0;

//~ private:
//~ /**
//~ * a map containing the list of scalar field available depending on
//~ * the physics enabled.
//~ */
//~ int2str_t index2names;

//~ /**
//~ * a map containing the list of scalar field available depending on
//~ * the physics enabled.
//~ */
//~ str2int_t names2index;

//~ /**
//~ * a Kokkos::Array to map a ComponentIndex to the actual index used in
//~ * LBMArray.
//~ */
//~ id2index_t id2index;

//~ public:
//~ /**
//~ * Initialize the Field Manager data (id2names and names2id)
//~ * using the configMap information.
//~ */
//~ void setup(const ConfigMap& configMap) {
//~ str2int_t names2dAll = get_names2id_all();

//~ const int nbComponent = COMPONENT_SIZE;

//~ /*
//~ * step1 : build a list of enabled variables, default is all false
//~ */
//~ std::array<int,nbComponent> var_enabled;
//~ std::string problem_variables = configMap.getString("lbm", "problem_variables", "rho");

//~ std::istringstream iss(problem_variables);
//~ std::string token;
//~ while (std::getline(iss, token, ',')) {
//~ var_enabled[names2dAll[token]] = 1;
//~ }

//~ /*
//~ * step2 : fill id2index map array.
//~ */

//~ // build a list of index mapped to a Component Index
//~ // if variabled is enabled, a unique index is attributed
//~ // else invalid index -1 is given
//~ int count = 0;
//~ for (int id=0; id<nbComponent; ++id) {

//~ if (var_enabled[id] == 1) {
//~ id2index[id] = count;
//~ count++;
//~ } else {
//~ id2index[id] = -1;
//~ }

//~ } // end for

//~ numScalarField = count;

//~ /*
//~ * init indexd2names (numScalarField elements),
//~ * unordered map of runtime available scalar field
//~ */
//~ int2str_t id2namesAll  = get_id2names_all();

//~ for (int id=0; id<nbComponent; ++id) {
//~ if (var_enabled[id] == 1) {
//~ // insert couple  ( index(id), name )
//~ index2names[ id2index[id] ] = id2namesAll[id];
//~ }
//~ }

//~ /*
//~ * init names2index (numScalarField elements),
//~ * unordered map of runtime available scalar field
//~ */

//~ for (int id=0; id<nbComponent; ++id) {
//~ if (var_enabled[id] == 1) {
//~ // insert couple  ( index(id), name )
//~ names2index[ id2namesAll[id] ] = id2index[id];
//~ }
//~ }

//~ } // setup

//~ int2str_t get_index2names() { return index2names; }

//~ str2int_t get_names2index() { return names2index; }

//~ id2index_t get_id2index() { return id2index; };

//~ /**
//~ * Builds an unordered_map between enum ComponentIndex and names (string)
//~ * using all available fields.
//~ *
//~ * \return map of id to names
//~ */
//~ static int2str_t get_id2names_all()
//~ {

//~ int2str_t map;

//~ // insert some fields
//~ map[ID]   = "rho";
//~ map[IU]   = "vx";
//~ map[IV]   = "vy";
//~ map[IW]   = "vz";
//~ map[IPHI] = "phi";
//~ map[IP]   = "pressure";
//~ map[IC]   = "composition";
//~ map[IMU]  = "mu";
//~ map[ICB]   = "cB";
//~ map[IMUB]  = "muB";
//~ map[IDPHIDX] = "dphidx";
//~ map[IDPHIDY] = "dphidy";
//~ map[IDPHIDZ] = "dphidz";
//~ map[ILAPLAPHI] = "laplaphi";
//~ map[IDPHIDT] = "dphidt";
//~ map[IDMUDT] = "dmudt";
//~ map[ISUPERSAT] = "supersat";
//~ map[IGRANDPOTENTIAL] = "grand_potential";
//~ map[IHELMOLTZENERGY] = "helmoltz_energy";
//~ map[IDCADX] = "dcadx";
//~ map[IDCADY] = "dcady";
//~ map[IDCADZ] = "dcadz";
//~ map[IDCBDX] = "dcbdx";
//~ map[IDCBDY] = "dcbdy";
//~ map[IDCBDZ] = "dcbdz";
//~ return map;

//~ } // get_id2names_all

//~ /**
//~ * Builds an unordered_map between fields names (string) and enum
//~ * ComponentIndex id, using all available fields.
//~ *
//~ * \return map of names to id
//~ */
//~ static str2int_t get_names2id_all()
//~ {

//~ str2int_t map;

//~ // insert some fields
//~ map["rho"]      = ID;
//~ map["vx"]       = IU;
//~ map["vy"]       = IV;
//~ map["vz"]       = IW;
//~ map["phi"]      = IPHI;
//~ map["pressure"] = IP;
//~ map["composition"] = IC;
//~ map["mu"]  = IMU;
//~ map["cB"] = ICB;
//~ map["muB"]  = IMUB;
//~ map["dphidx"] = IDPHIDX;
//~ map["dphidy"] = IDPHIDY;
//~ map["dphidz"] = IDPHIDZ;
//~ map["laplaphi"] = ILAPLAPHI;
//~ map["dphidt"] = IDPHIDT;
//~ map["dmudt"] = IDMUDT;
//~ map["supersat"] = ISUPERSAT;
//~ map["grand_potential"] = IGRANDPOTENTIAL;
//~ map["helmoltz_energy"] = IHELMOLTZENERGY;
//~ map["dcadx"] = IDCADX;
//~ map["dcady"] = IDCADY;
//~ map["dcadz"] = IDCADZ;
//~ map["dcbdx"] = IDCBDX;
//~ map["dcbdy"] = IDCBDY;
//~ map["dcbdz"] = IDCBDZ;

//~ return map;

//~ } // get_names2id_all

//~ }; // FieldManager

#endif // FIELD_MANAGER_H_
