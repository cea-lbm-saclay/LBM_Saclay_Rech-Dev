#include "IO_ReadWrite.h"

#include <LBMParams.h>
#include <utils/config/ConfigMap.h>

#include "save_vtk.h"

#ifdef USE_HDF5
#include "IO_HDF5.h"
#endif

// #ifdef USE_PNETCDF
// #include "IO_PNETCDF.h"
// #endif // USE_PNETCDF

namespace lbm_saclay {

// =======================================================
// =======================================================
IO_ReadWrite::IO_ReadWrite(LBMParams& params,
    ConfigMap& configMap,
    int2str_t& index2names)
    : params(params)
    , configMap(configMap)
    , index2names(index2names)
    , vtk_enabled(true)
    , hdf5_enabled(false)
{

    // do we want VTK output ?
    vtk_enabled = configMap.getBool("output", "vtk_enabled", true);

    // do we want HDF5 output ?
    hdf5_enabled = configMap.getBool("output", "hdf5_enabled", false);

    // do we want Parallel NETCDF output ? Only valid/activated for MPI run
    //pnetcdf_enabled = configMap.getBool("output","pnetcdf_enabled", false);

} // IO_ReadWrite::IO_ReadWrite

// =======================================================
// =======================================================
void IO_ReadWrite::save_data(LBMArray2d data_d,
    LBMArray2dHost data_h,
    int iStep,
    real_t time,
    int m_times_saved,
    int2str_t& var2write)
{

    save_data_impl(data_d, data_h, iStep, time, m_times_saved, var2write);

} // IO_ReadWrite::save_data

// =======================================================
// =======================================================
void IO_ReadWrite::save_data(LBMArray3d data_d,
    LBMArray3dHost data_h,
    int iStep,
    real_t time,
    int m_times_saved,
    int2str_t& var2write)
{

    save_data_impl(data_d, data_h, iStep, time, m_times_saved, var2write);

} // IO_ReadWrite::save_data

// =======================================================
// =======================================================
void IO_ReadWrite::save_data_impl(LBMArray2d data_d,
    LBMArray2dHost data_h,
    int iStep,
    real_t time,
    int m_times_saved,
    int2str_t& var2write)
{

    // copy device data to host data
    Kokkos::deep_copy(data_h, data_d);

    if (vtk_enabled) {

#ifdef USE_MPI
        lbm_saclay::save_vtk_mpi(data_h, params, configMap, var2write, iStep);
#else
        lbm_saclay::save_vtk(data_h, params, configMap, var2write, iStep);
#endif // USE_MPI
    }

#ifdef USE_HDF5
    if (hdf5_enabled) {

#ifdef USE_MPI
        lbm_saclay::Save_HDF5_mpi<TWO_D> writer(data_h, params, configMap, var2write, iStep, time, m_times_saved);
        writer.save();
#else
        lbm_saclay::Save_HDF5<TWO_D> writer(data_h, params, configMap, var2write, iStep, time, m_times_saved);
        writer.save();
#endif // USE_MPI
    }
#endif // USE_HDF5

    // #ifdef USE_PNETCDF
    // 	if (pnetcdf_enabled) {
    // 		lbm_saclay::Save_PNETCDF<TWO_D> writer(data_d, data_h, params, configMap, HYDRO_2D_NBVAR, variables_names, iStep, time, debug_name);
    // 		writer.save();
    // 	}
    // #endif // USE_PNETCDF

} // IO_ReadWrite::save_data_impl

// =======================================================
// =======================================================
void IO_ReadWrite::save_data_impl(LBMArray3d data_d,
    LBMArray3dHost data_h,
    int iStep,
    real_t time,
    int m_times_saved,
    int2str_t& var2write)
{
    // copy device data to host data
    Kokkos::deep_copy(data_h, data_d);

    if (vtk_enabled) {

#ifdef USE_MPI
        lbm_saclay::save_vtk_mpi(data_h, params, configMap, var2write, iStep);
#else
        lbm_saclay::save_vtk(data_h, params, configMap, var2write, iStep);
#endif // USE_MPI
    }

#ifdef USE_HDF5
    if (hdf5_enabled) {

#ifdef USE_MPI
        lbm_saclay::Save_HDF5_mpi<THREE_D> writer(data_h, params, configMap, var2write, iStep, time, m_times_saved);
        writer.save();
#else
        lbm_saclay::Save_HDF5<THREE_D> writer(data_h, params, configMap, var2write, iStep, time, m_times_saved);
        writer.save();
#endif // USE_MPI
    }
#endif // USE_HDF5

    // #ifdef USE_PNETCDF
    // 	if (pnetcdf_enabled) {
    // 		lbm_saclay::Save_PNETCDF<THREE_D> writer(data_d, data_h, params, configMap, HYDRO_2D_NBVAR, variables_names, iStep, time, debug_name);
    // 		writer.save();
    // 	}
    // #endif // USE_PNETCDF

} // IO_ReadWrite::save_data_impl

// =======================================================
// =======================================================
void IO_ReadWrite::load_data(LBMArray2d data_d,
    LBMArray2dHost data_h,
    int& iStep,
    real_t& time,
    int& m_times_saved)
{

    load_data_impl(data_d, data_h, iStep, time, m_times_saved);

} // IO_ReadWrite::save_data

// =======================================================
// =======================================================
void IO_ReadWrite::load_data(LBMArray3d data_d,
    LBMArray3dHost data_h,
    int& iStep,
    real_t& time,
    int& m_times_saved)
{

    load_data_impl(data_d, data_h, iStep, time, m_times_saved);

} // IO_ReadWrite::load_data

// =======================================================
// =======================================================
void IO_ReadWrite::load_data_impl(LBMArray2d data_d,
    LBMArray2dHost data_h,
    int& iStep,
    real_t& time,
    int& m_times_saved)
{

    //! get input filename from configMap
    std::string inputFilename = configMap.getString("init", "restart_filename", "");
    std::string inputDir = configMap.getString("init", "inputDir", ".");
    std::string completeFilename = inputDir + "/" + inputFilename;

    //!check filename extension
    bool isHdf5 = false;
    bool isNcdf = false;
    detectFileType(isHdf5, isNcdf, inputFilename);

#ifdef USE_HDF5
    if (hdf5_enabled and isHdf5) {

#ifdef USE_MPI
        lbm_saclay::Load_HDF5_mpi<TWO_D> reader(data_h, params, configMap, index2names);
        reader.load(completeFilename);
        Kokkos::deep_copy(data_d, data_h);
#else
        lbm_saclay::Load_HDF5<TWO_D> reader(data_h, params, configMap, index2names);
        reader.load(completeFilename);
        Kokkos::deep_copy(data_d, data_h);
#endif // USE_MPI

        // get time information from reader
        iStep = reader.iStep;
        time = reader.totalTime;
        m_times_saved = reader.m_times_saved;
    }
#endif // USE_HDF5

} // IO_ReadWrite::load_data_impl - 2d

// =======================================================
// =======================================================
void IO_ReadWrite::load_data_impl(LBMArray3d data_d,
    LBMArray3dHost data_h,
    int& iStep,
    real_t& time,
    int& m_times_saved)
{

    // get input filename from configMap
    std::string inputFilename = configMap.getString("init", "restart_filename", "");
    std::string inputDir = configMap.getString("init", "inputDir", "./");
    std::string completeFilename = inputDir + "/" + inputFilename;

    // check filename extension
    bool isHdf5 = false;
    bool isNcdf = false;
    detectFileType(isHdf5, isNcdf, inputFilename);

#ifdef USE_HDF5
    if (hdf5_enabled and isHdf5) {

#ifdef USE_MPI
        lbm_saclay::Load_HDF5_mpi<THREE_D> reader(data_h, params, configMap, index2names);
        reader.load(inputFilename);
        Kokkos::deep_copy(data_d, data_h);
#else
        lbm_saclay::Load_HDF5<THREE_D> reader(data_h, params, configMap, index2names);
        reader.load(inputFilename);
        Kokkos::deep_copy(data_d, data_h);
#endif // USE_MPI

        iStep = reader.iStep;
        time = reader.totalTime;
        m_times_saved = reader.m_times_saved;
    }
#endif // USE_HDF5

} // IO_ReadWrite::load_data_impl - 3d

void IO_ReadWrite::detectFileType(bool& isHdf5, bool& isNcdf, std::string inputFilename)
{

    // check filename extension
    std::string h5Suffix(".h5");
    std::string ncSuffix(".nc"); // pnetcdf file only available when MPI is activated
    if (inputFilename.length() >= 3) {
        isHdf5 = (0 == inputFilename.compare(inputFilename.length() - h5Suffix.length(), h5Suffix.length(), h5Suffix));
        isNcdf = (0 == inputFilename.compare(inputFilename.length() - ncSuffix.length(), ncSuffix.length(), ncSuffix));
    }
} // IO_ReadWrite::detectFileType

} // namespace lbm_saclay
