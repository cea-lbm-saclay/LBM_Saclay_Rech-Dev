#ifndef SAVE_VTK_H_
#define SAVE_VTK_H_

#include <map>

#include "FieldManager.h"
#include "LBMParams.h"
#include "kokkos_shared.h"
#include "utils/config/ConfigMap.h"

namespace lbm_saclay {

/**
 * Host routine to save data to file (vti = VTK image format).
 *
 * about VTK file format: ASCII, VtkImageData
 * Take care that VTK uses row major (i+j*nx)
 * To make sure OpenMP and CUDA version give the same
 * results, we transpose the OpenMP data.
 *
 * \param[in,out] data_h a host LBMArray used as a temporary storage (allocated by the caller to avoid allocating inside this routine)
 * \param[in] params a LBMParams reference object (input only)
 * \param[in] configMap a ConfigMap reference object (input only)
 * \param[in] iStep integer, current time step used in filename
 * \param[in] name as string
 *
 */
void save_vtk(LBMArray2dHost data_h,
    const LBMParams& params,
    const ConfigMap& configMap,
    const int2str_t& index2names,
    int iStep);

void save_vtk(LBMArray3dHost data_h,
    const LBMParams& params,
    const ConfigMap& configMap,
    const int2str_t& index2names,
    int iStep);

#ifdef USE_MPI
/**
 * Host routine to save data to file using (pvti = partitionned VTK
 * image format).
 *
 * One vti file per MPI process is written; an additional pvti file (xml like)
 * is written to gather them all and make paraview load them and recompose the
 * entire data set.
 */
void save_vtk_mpi(LBMArray2dHost data_h,
    const LBMParams& params,
    const ConfigMap& configMap,
    const int2str_t& index2name,
    int iStep);

void save_vtk_mpi(LBMArray3dHost data_h,
    const LBMParams& params,
    const ConfigMap& configMap,
    const int2str_t& index2name,
    int iStep);

/**
 * Write Parallel VTI header.
 * Must be done by a single MPI process, usually root process.
 *
 */
void write_pvti_header(std::string headerFilename,
    std::string outputPrefix,
    const DimensionType dim,
    const LBMParams& params,
    const int2str_t& index2names,
    int iStep);

#endif // USE_MPI

} // namespace lbm_saclay

#endif // SAVE_VTK_H_
