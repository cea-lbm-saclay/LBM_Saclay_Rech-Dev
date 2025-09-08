#include "IO_HDF5.h"

#include "utils/config/ConfigMap.h"

#include <fstream>

namespace lbm_saclay {

// =======================================================
// =======================================================
void writeXdmfForHdf5Wrapper(LBMParams& params,
    ConfigMap& configMap,
    const int2str_t& index2names,
    int totalNumberOfSteps,
    bool singleStep)
{

    // domain (no-MPI) or sub-domain sizes (MPI)
    const int nx = params.nx;
    const int ny = params.ny;
    const int nz = params.nz;

    const real_t dx = params.dx;
    const real_t dy = params.dy;
    const real_t dz = params.dz;

#ifdef USE_MPI
    // sub-domain decomposition sizes
    const int mx = params.mx;
    const int my = params.my;
    const int mz = params.mz;
#endif

    const int ghostWidth = params.ghostWidth;

    const int dimType = params.dimType;

    const bool ghostIncluded = configMap.getBool("output", "ghostIncluded", false);

#ifdef USE_MPI
    // global sizes
    int nxg = mx * nx;
    int nyg = my * ny;
    int nzg = mz * nz;
#else
    // data size actually written on disk
    int nxg = nx;
    int nyg = ny;
    int nzg = nz;
#endif // USE_MPI

    if (ghostIncluded) {
        nxg += (2 * ghostWidth);
        nyg += (2 * ghostWidth);
        nzg += (2 * ghostWidth);
    }

#ifdef USE_MPI

    /*
	 * Let MPIIO underneath hdf5 re-assemble the pieces and provides a single
	 * nice file. Thanks parallel HDF5 !
	 */

#endif // USE_MPI

    // get data type as a string for Xdmf
    std::string dataTypeName;
    if (sizeof(real_t) == sizeof(float))
        dataTypeName = "Float";
    else
        dataTypeName = "Double";

    /*
	 * 1. open XDMF and write header lines
	 */
    std::string outputDir = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
    std::string xdmfFilename = outputDir + "/" + outputPrefix + ".xmf";

    if (singleStep) {
        // add iStep to file name
        auto outNum = format_index(totalNumberOfSteps, 7);
        xdmfFilename = outputDir + "/" + outputPrefix + "_" + outNum + ".xmf";
    }
    std::fstream xdmfFile;
    xdmfFile.open(xdmfFilename.c_str(), std::ios_base::out);

    xdmfFile << "<?xml version=\"1.0\" ?>" << std::endl;
    xdmfFile << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>" << std::endl;
    xdmfFile << "<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2.2\">" << std::endl;
    xdmfFile << "  <Domain>" << std::endl;
    xdmfFile << "    <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">" << std::endl;

    // for each time step write a <grid> </grid> item
    int startStep = 0;
    int stopStep = totalNumberOfSteps * params.nOutput;
    int deltaStep = params.nOutput;
    if (params.nOutput == -1) {
        stopStep = totalNumberOfSteps;
        deltaStep = 1;
    }

    if (singleStep) {
        startStep = totalNumberOfSteps;
        stopStep = totalNumberOfSteps + 1;
        deltaStep = 1;
    }

    for (int iStep = startStep; iStep <= stopStep; iStep += deltaStep) {

        auto outNum = format_index(iStep, 7);

        // take care that the following filename must be exactly the same as in routine outputHdf5 !!!
        std::string baseName = outputPrefix + "_" + outNum;
        std::string hdf5Filename = outputPrefix + "_" + outNum + ".h5";
        std::string hdf5FilenameFull = outputDir + "/" + outputPrefix + "_" + outNum + ".h5";

        xdmfFile << "    <Grid Name=\"" << baseName << "\" GridType=\"Uniform\">" << std::endl;
        xdmfFile << "    <Time Value=\"" << iStep << "\" />" << std::endl;

        // topology CoRectMesh
        if (dimType == TWO_D)
            xdmfFile << "      <Topology TopologyType=\"2DCoRectMesh\" NumberOfElements=\"" << nyg + 1 << " " << nxg + 1 << "\"/>" << std::endl;
        else
            xdmfFile << "      <Topology TopologyType=\"3DCoRectMesh\" NumberOfElements=\"" << nzg + 1 << " " << nyg + 1 << " " << nxg + 1 << "\"/>" << std::endl;

        // geometry
        if (dimType == TWO_D) {
            xdmfFile << "    <Geometry Type=\"ORIGIN_DXDY\">" << std::endl;
            xdmfFile << "    <DataStructure" << std::endl;
            xdmfFile << "       Name=\"Origin\"" << std::endl;
            xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
            xdmfFile << "       Dimensions=\"2\"" << std::endl;
            xdmfFile << "       Format=\"XML\">" << std::endl;
            xdmfFile << "       0 0" << std::endl;
            xdmfFile << "    </DataStructure>" << std::endl;
            xdmfFile << "    <DataStructure" << std::endl;
            xdmfFile << "       Name=\"Spacing\"" << std::endl;
            xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
            xdmfFile << "       Dimensions=\"2\"" << std::endl;
            xdmfFile << "       Format=\"XML\">" << std::endl;
            xdmfFile << "       " << dy << " " << dx << std::endl;
            xdmfFile << "    </DataStructure>" << std::endl;
            xdmfFile << "    </Geometry>" << std::endl;
        } else {
            xdmfFile << "    <Geometry Type=\"ORIGIN_DXDYDZ\">" << std::endl;
            xdmfFile << "    <DataStructure" << std::endl;
            xdmfFile << "       Name=\"Origin\"" << std::endl;
            xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
            xdmfFile << "       Dimensions=\"3\"" << std::endl;
            xdmfFile << "       Format=\"XML\">" << std::endl;
            xdmfFile << "       0 0 0" << std::endl;
            xdmfFile << "    </DataStructure>" << std::endl;
            xdmfFile << "    <DataStructure" << std::endl;
            xdmfFile << "       Name=\"Spacing\"" << std::endl;
            xdmfFile << "       DataType=\"" << dataTypeName << "\"" << std::endl;
            xdmfFile << "       Dimensions=\"3\"" << std::endl;
            xdmfFile << "       Format=\"XML\">" << std::endl;
            xdmfFile << "       " << dz << " " << dy << " " << dx << std::endl;
            xdmfFile << "    </DataStructure>" << std::endl;
            xdmfFile << "    </Geometry>" << std::endl;
        }

        for (auto iter : index2names) {

            // get variables string name
            const std::string varName = iter.second;

            // get variable id
            //int iVar = iter.second;

            xdmfFile << "      <Attribute Center=\"Cell\" Name=\"" << varName << "\">" << std::endl;
            xdmfFile << "        <DataStructure" << std::endl;
            xdmfFile << "           DataType=\"" << dataTypeName << "\"" << std::endl;
            if (dimType == TWO_D)
                xdmfFile << "           Dimensions=\"" << nyg << " " << nxg << "\"" << std::endl;
            else
                xdmfFile << "           Dimensions=\"" << nzg << " " << nyg << " " << nxg << "\"" << std::endl;
            xdmfFile << "           Format=\"HDF\">" << std::endl;
            xdmfFile << "           " << hdf5Filename << ":/" << varName << "" << std::endl;
            xdmfFile << "        </DataStructure>" << std::endl;
            xdmfFile << "      </Attribute>" << std::endl;
        }

        // finalize grid file for the current time step
        xdmfFile << "   </Grid>" << std::endl;

    } // end for loop over time step

    // finalize Xdmf wrapper file
    xdmfFile << "   </Grid>" << std::endl;
    xdmfFile << " </Domain>" << std::endl;
    xdmfFile << "</Xdmf>" << std::endl;

} // writeXdmfForHdf5Wrapper

} // namespace lbm_saclay
