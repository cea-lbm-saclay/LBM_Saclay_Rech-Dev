#ifndef LBM_SACLAY_IO_READ_WRITE_H_
#define LBM_SACLAY_IO_READ_WRITE_H_

#include <map>
#include <string>

#include "LBMParams.h"
#include "io/io_common.h"
#include "kokkos_shared.h"
#include "utils/config/ConfigMap.h"

namespace lbm_saclay {

/**
 *
 */
class IO_ReadWrite {

public:
    IO_ReadWrite(LBMParams& params,
        ConfigMap& configMap,
        int2str_t& index2names);

    //! destructor
    ~IO_ReadWrite() = default;

    //! LBM parameters
    LBMParams& params;

    //! configuration file reader
    ConfigMap& configMap;

    //! names of variables to load/save (inherited from Solver)
    int2str_t index2names;

    bool vtk_enabled;
    bool hdf5_enabled;
    //~ bool pnetcdf_enabled;

    //! override base class method
    void save_data(LBMArray2d data_d,
        LBMArray2dHost data_h,
        int iStep,
        real_t time,
        int m_times_saved,
        int2str_t& var2write);

    //! override base class method
    void save_data(LBMArray3d data_d,
        LBMArray3dHost data_h,
        int iStep,
        real_t time,
        int m_times_saved,
        int2str_t& var2write);

    //! public interface to save data.
    void save_data_impl(LBMArray2d data_d,
        LBMArray2dHost data_h,
        int iStep,
        real_t time,
        int m_times_saved,
        int2str_t& var2write);

    void save_data_impl(LBMArray3d data_d,
        LBMArray3dHost data_h,
        int iStep,
        real_t time,
        int m_times_saved,
        int2str_t& var2write);

    //! override base class method
    void load_data(LBMArray2d data_d,
        LBMArray2dHost data_h,
        int& iStep,
        real_t& time,
        int& m_times_saved);

    //! override base class method
    void load_data(LBMArray3d data_d,
        LBMArray3dHost data_h,
        int& iStep,
        real_t& time,
        int& m_times_saved);

    //! public interface to load data.
    void load_data_impl(LBMArray2d data_d,
        LBMArray2dHost data_h,
        int& iStep,
        real_t& time,
        int& m_times_saved);

    void load_data_impl(LBMArray3d data_d,
        LBMArray3dHost data_h,
        int& iStep,
        real_t& time,
        int& m_times_saved);

    void detectFileType(bool& isHdf5, bool& isNcdf, std::string inputFilename);

}; // class IO_ReadWrite

} // namespace lbm_saclay

#endif // LBM_SACLAY_IO_READ_WRITE_H_
