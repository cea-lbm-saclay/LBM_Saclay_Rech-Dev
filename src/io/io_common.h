#ifndef LBM_SACLAY_IO_COMMON_H_
#define LBM_SACLAY_IO_COMMON_H_

#include "FieldManager.h"
#include "LBMParams.h"
#include "kokkos_shared.h"
#include "utils/config/ConfigMap.h"

#include <string>

namespace lbm_saclay {

// =======================================================
// =======================================================
/**
 * Format an integer into a string of a given lenght (padding with zeros in front).
 *
 * \param[in] index integer to convert
 * \param[in] fmt_width length of the formatted string
 *
 * \return formatted string
 */
std::string format_index(const int index, const int fmt_width);

// =======================================================
// =======================================================
/**
 * Return current date in a string.
 */
std::string current_date();

// =======================================================
// =======================================================
/**
 * Use configMap information to retrieve a list of scalar field to write,
 * compute the their id the access them in LBMArray. This routine returns
 * a map with this information.
 *
 * \param[in,out] map this is the map to fill
 * \param[in] params a LBMParams reference object
 * \param[in] configMap to access parameters settings
 *
 * \return the map size (i.e. the number of valid variable names)
 */
//~ int build_var_to_write_map(str2int_t& map,
//~ const LBMParams& params,
//~ const ConfigMap& configMap);

int2str_t build_var_to_write_map(const LBMParams& params, const ConfigMap& configMap, int2str_t avail_ids);

} // namespace lbm_saclay

#endif // LBM_SACLAY_IO_COMMON_H_
