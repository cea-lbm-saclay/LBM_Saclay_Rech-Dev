#include "io/save_vtk.h"

#include "io/io_common.h"

// for string tokenizer
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace lbm_saclay {

static bool isBigEndian()
{
    const int i = 1;
    return ((*(char*)&i) == 0);
}

// =======================================================
// =======================================================
void save_vtk(LBMArray2dHost data_h,
    const LBMParams& params,
    const ConfigMap& configMap,
    const int2str_t& index2names,
    int iStep)
{

    // some useful parameter alias
    const int nx = params.nx;
    const int ny = params.ny;

    const real_t dx = params.dx;
    const real_t dy = params.dy;

    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;

    const int imin = params.imin;
    const int imax = params.imax;

    const int jmin = params.jmin;
    const int jmax = params.jmax;

    const int ghostWidth = params.ghostWidth;

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int nbCells = isize * jsize;

    // local variables
    int i, j;
    std::string outputDir = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

    bool isFinal = (iStep * params.dt >= params.tEnd or iStep >= params.nStepmax);

    bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);

    // check scalar data type
    bool useDouble = false;

    if (sizeof(real_t) == sizeof(double)) {
        useDouble = true;
    }

    // write iStep in string stepNum
    auto stepNum = format_index(iStep, 10);

    std::string filename;
    // concatenate file prefix + file number + suffix
    if (isFinal) {
        filename = outputDir + "/" + outputPrefix + "_FINAL" + ".vti";
    } else {
        filename = outputDir + "/" + outputPrefix + "_" + stepNum + ".vti";
    }

    // open file
    std::fstream outFile;
    outFile.open(filename.c_str(), std::ios_base::out);

    // write header

    // if writing raw binary data (file does not respect XML standard)
    if (outputVtkAscii)
        outFile << "<?xml version=\"1.0\"?>\n";

    // write xml data header
    if (isBigEndian()) {
        outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"BigEndian\">\n";
    } else {
        outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    }

    // write mesh extent
    outFile << "  <ImageData WholeExtent=\""
            << 0 << " " << nx << " "
            << 0 << " " << ny << " "
            << 0 << " " << 0 << "\" "
            << "Origin=\""
            << xmin << " " << ymin << " " << 0.0 << "\" "
            << "Spacing=\""
            << dx << " " << dy << " " << 0.0 << "\">\n";
    outFile << "  <Piece Extent=\""
            << 0 << " " << nx << " "
            << 0 << " " << ny << " "
            << 0 << " " << 0 << " "
            << "\">\n";

    outFile << "    <PointData>\n";
    outFile << "    </PointData>\n";

    if (outputVtkAscii) {

        outFile << "    <CellData>\n";

        // write data array (ascii), remove ghost cells
        for (auto& iter : index2names) {

            // get variable id
            int iVar = iter.first;

            // get variables string name
            const std::string varName = iter.second;

            outFile << "    <DataArray type=\"";
            if (useDouble) {
                outFile << "Float64";
            } else {
                outFile << "Float32";
            }
            outFile << "\" Name=\"" << varName << "\" format=\"ascii\" >\n";

            for (int index = 0; index < nbCells; ++index) {

                // enforce the use of left layout (Ok for CUDA)
                // but for OpenMP, we will need to transpose
                j = index / isize;
                i = index - j * isize;

                if (j >= jmin + ghostWidth and j <= jmax - ghostWidth and i >= imin + ghostWidth and i <= imax - ghostWidth) {
                    outFile << data_h(i, j, iVar) << " ";
                }
            }
            outFile << "\n    </DataArray>\n";
        } // end for iVar

        outFile << "    </CellData>\n";

        // write footer
        outFile << "  </Piece>\n";
        outFile << "  </ImageData>\n";
        outFile << "</VTKFile>\n";

    } else { // write data in binary format

        outFile << "    <CellData>" << std::endl;

        int idVar = 0;
        for (auto& iter : index2names) {

            // get variable id
            //int iVar = iter.first;

            // get variables string name
            const std::string varName = iter.second;

            if (useDouble) {
                outFile << "     <DataArray type=\"Float64\" Name=\"";
            } else {
                outFile << "     <DataArray type=\"Float32\" Name=\"";
            }
            outFile << varName
                    << "\" format=\"appended\" offset=\""
                    << idVar * nx * ny * sizeof(real_t) + idVar * sizeof(unsigned int)
                    << "\" />" << std::endl;

            idVar++;
        } // end for names2index

        outFile << "    </CellData>" << std::endl;
        outFile << "  </Piece>" << std::endl;
        outFile << "  </ImageData>" << std::endl;

        outFile << "  <AppendedData encoding=\"raw\">" << std::endl;

        // write the leading undescore
        outFile << "_";
        // then write heavy data (column major format)
        {
            unsigned int nbOfWords = nx * ny * sizeof(real_t);
            for (auto iter : index2names) {

                // get variable id
                int iVar = iter.first;

                outFile.write((char*)&nbOfWords, sizeof(unsigned int));
                for (int j = jmin + ghostWidth; j <= jmax - ghostWidth; j++)
                    for (int i = imin + ghostWidth; i <= imax - ghostWidth; i++) {
                        real_t tmp = data_h(i, j, iVar);
                        outFile.write((char*)&tmp, sizeof(real_t));
                    }
            } // end for names2index
        }

        outFile << "  </AppendedData>" << std::endl;
        outFile << "</VTKFile>" << std::endl;

    } // end ascii/binary heavy data write

    outFile.close();

} // save_vtk

// =======================================================
// =======================================================
void save_vtk(LBMArray3dHost data_h,
    const LBMParams& params,
    const ConfigMap& configMap,
    const int2str_t& index2names,
    int iStep)
{

    // some useful parameter alias
    const int nx = params.nx;
    const int ny = params.ny;
    const int nz = params.nz;

    const real_t dx = params.dx;
    const real_t dy = params.dy;
    const real_t dz = params.dz;

    const real_t xmin = params.xmin;
    const real_t ymin = params.ymin;
    const real_t zmin = params.zmin;

    const int imin = params.imin;
    const int imax = params.imax;

    const int jmin = params.jmin;
    const int jmax = params.jmax;

    const int kmin = params.kmin;
    const int kmax = params.kmax;

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int nbCells = isize * jsize * ksize;
    const int ijsize = isize * jsize;

    const int ghostWidth = params.ghostWidth;

    // local variables
    int i, j, k;
    std::string outputDir = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

    bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);

    // check scalar data type
    bool useDouble = false;

    if (sizeof(real_t) == sizeof(double)) {
        useDouble = true;
    }

    // write iStep in string stepNum
    auto stepNum = format_index(iStep, 7);

    // concatenate file prefix + file number + suffix
    std::string filename = outputDir + "/" + outputPrefix + "_" + stepNum + ".vti";

    // open file
    std::fstream outFile;
    outFile.open(filename.c_str(), std::ios_base::out);

    // write header

    // if writing raw binary data (file does not respect XML standard)
    if (outputVtkAscii)
        outFile << "<?xml version=\"1.0\"?>\n";

    if (isBigEndian()) {
        outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"BigEndian\">\n";
    } else {
        outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    }

    // write mesh extent
    outFile << "  <ImageData WholeExtent=\""
            << 0 << " " << nx << " "
            << 0 << " " << ny << " "
            << 0 << " " << nz << "\" "
            << "Origin=\""
            << xmin << " " << ymin << " " << zmin << "\" "
            << "Spacing=\""
            << dx << " " << dy << " " << dz << "\">\n";
    outFile << "  <Piece Extent=\""
            << 0 << " " << nx << " "
            << 0 << " " << ny << " "
            << 0 << " " << nz << " "
            << "\">\n";

    outFile << "    <PointData>\n";
    outFile << "    </PointData>\n";

    if (outputVtkAscii) {

        outFile << "    <CellData>\n";

        // write data array (ascii), remove ghost cells
        for (auto& iter : index2names) {

            // get variable id
            int iVar = iter.first;

            // get variables string name
            const std::string varName = iter.second;

            outFile << "    <DataArray type=\"";
            if (useDouble)
                outFile << "Float64";
            else
                outFile << "Float32";
            outFile << "\" Name=\"" << varName << "\" format=\"ascii\" >\n";

            for (int index = 0; index < nbCells; ++index) {

                // enforce the use of left layout (Ok for CUDA)
                // but for OpenMP, we will need to transpose
                k = index / ijsize;
                j = (index - k * ijsize) / isize;
                i = index - j * isize - k * ijsize;

                if (k >= kmin + ghostWidth and k <= kmax - ghostWidth and j >= jmin + ghostWidth and j <= jmax - ghostWidth and i >= imin + ghostWidth and i <= imax - ghostWidth) {
                    outFile << data_h(i, j, k, iVar) << " ";
                }
            }
            outFile << "\n    </DataArray>\n";
        } // end for iVar

        outFile << "    </CellData>\n";

        // write footer
        outFile << "  </Piece>\n";
        outFile << "  </ImageData>\n";
        outFile << "</VTKFile>\n";

    } else { // write data in binary format

        outFile << "    <CellData>" << std::endl;

        int idVar = 0;
        for (auto& iter : index2names) {

            // get variable id
            //int iVar = iter.first;

            // get variables string name
            const std::string varName = iter.second;

            if (useDouble) {
                outFile << "     <DataArray type=\"Float64\" Name=\"";
            } else {
                outFile << "     <DataArray type=\"Float32\" Name=\"";
            }
            outFile << varName
                    << "\" format=\"appended\" offset=\""
                    << idVar * nx * ny * nz * sizeof(real_t) + idVar * sizeof(unsigned int)
                    << "\" />" << std::endl;

            idVar++;

        } // end for names2index

        outFile << "    </CellData>" << std::endl;
        outFile << "  </Piece>" << std::endl;
        outFile << "  </ImageData>" << std::endl;

        outFile << "  <AppendedData encoding=\"raw\">" << std::endl;

        // write the leading undescore
        outFile << "_";

        // then write heavy data (column major format)
        {
            unsigned int nbOfWords = nx * ny * nz * sizeof(real_t);
            for (auto& iter : index2names) {

                // get variable id
                int iVar = iter.first;

                outFile.write((char*)&nbOfWords, sizeof(unsigned int));
                for (int k = kmin + ghostWidth; k <= kmax - ghostWidth; k++)
                    for (int j = jmin + ghostWidth; j <= jmax - ghostWidth; j++)
                        for (int i = imin + ghostWidth; i <= imax - ghostWidth; i++) {
                            real_t tmp = data_h(i, j, k, iVar);
                            outFile.write((char*)&tmp, sizeof(real_t));
                        }
            } // end for names2index
        }

        outFile << "  </AppendedData>" << std::endl;
        outFile << "</VTKFile>" << std::endl;

    } // end ascii/binary heavy data write

    outFile.close();

} // save_vtk

#ifdef USE_MPI
// =======================================================
// =======================================================
void save_vtk_mpi(LBMArray2dHost data_h,
    const LBMParams& params,
    const ConfigMap& configMap,
    const int2str_t& index2names,
    int iStep)
{

    // some useful parameter alias
    const int nx = params.nx;
    const int ny = params.ny;

    const int imin = params.imin;
    const int imax = params.imax;

    const int jmin = params.jmin;
    const int jmax = params.jmax;

    const int ghostWidth = params.ghostWidth;

    const real_t dx = params.dx;
    const real_t dy = params.dy;
    //const real_t dz = 0.0;

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int nbCells = isize * jsize;

    int xmin = 0, xmax = 0, ymin = 0, ymax = 0;

    xmin = params.myMpiPos[0] * nx;
    xmax = params.myMpiPos[0] * nx + nx;
    ymin = params.myMpiPos[1] * ny;
    ymax = params.myMpiPos[1] * ny + ny;

    // local variables
    int i, j;
    std::string outputDir = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

    bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);

    // check scalar data type
    bool useDouble = false;

    if (sizeof(real_t) == sizeof(double)) {
        useDouble = true;
    }

    // write iStep in string stepNum
    auto stepNum = format_index(iStep, 7);

    // write MPI rank in string rankFormat
    auto rank = format_index(params.myRank, 5);

    // write pvti header file
    if (params.myRank == 0) {
        // header file : parallel vti format
        std::string headerFilename = outputDir + "/" + outputPrefix + "_time" + stepNum + ".pvti";

        write_pvti_header(headerFilename,
            outputPrefix,
            DIM2,
            params,
            index2names,
            iStep);
    }

    // concatenate file prefix + file number + suffix
    std::string filename = outputDir + "/" + outputPrefix + "_time" + stepNum + "_mpi" + rank + ".vti";

    // open file
    std::fstream outFile;
    outFile.open(filename.c_str(), std::ios_base::out);

    // write header

    // if writing raw binary data (file does not respect XML standard)
    if (outputVtkAscii)
        outFile << "<?xml version=\"1.0\"?>\n";

    if (isBigEndian()) {
        outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"BigEndian\">\n";
    } else {
        outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    }

    // write mesh extent
    outFile << "  <ImageData WholeExtent=\""
            << xmin << " " << xmax << " "
            << ymin << " " << ymax << " "
            << 0 << " " << 0 << "\" "
            << "Origin=\""
            << params.xmin << " " << params.ymin << " " << 0.0 << "\" "
            << "Spacing=\""
            << dx << " " << dy << " " << 0.0 << "\">\n";
    outFile << "  <Piece Extent=\""
            << xmin << " " << xmax << " "
            << ymin << " " << ymax << " "
            << 0 << " " << 0 << " "
            << "\">\n";

    outFile << "    <PointData>\n";
    outFile << "    </PointData>\n";

    if (outputVtkAscii) {

        outFile << "    <CellData>\n";

        // write data array (ascii), remove ghost cells
        for (auto& iter : index2names) {

            // get variable id
            int iVar = iter.first;

            // get variables string name
            const std::string varName = iter.second;

            outFile << "    <DataArray type=\"";
            if (useDouble) {
                outFile << "Float64";
            } else {
                outFile << "Float32";
            }
            outFile << "\" Name=\"" << varName << "\" format=\"ascii\" >\n";

            for (int index = 0; index < nbCells; ++index) {

                // enforce the use of left layout (Ok for CUDA)
                // but for OpenMP, we will need to transpose
                j = index / isize;
                i = index - j * isize;

                if (j >= jmin + ghostWidth and j <= jmax - ghostWidth and i >= imin + ghostWidth and i <= imax - ghostWidth) {
                    outFile << data_h(i, j, iVar) << " ";
                }
            }
            outFile << "\n    </DataArray>\n";
        } // end for iVar

        outFile << "    </CellData>\n";

        // write footer
        outFile << "  </Piece>\n";
        outFile << "  </ImageData>\n";
        outFile << "</VTKFile>\n";

    } else { // write data in binary format

        outFile << "    <CellData>" << std::endl;

        int idVar = 0;
        for (auto& iter : index2names) {

            // get variable id
            //int iVar = iter.first;

            // get variables string name
            const std::string varName = iter.second;

            if (useDouble) {
                outFile << "     <DataArray type=\"Float64\" Name=\"";
            } else {
                outFile << "     <DataArray type=\"Float32\" Name=\"";
            }
            outFile << varName
                    << "\" format=\"appended\" offset=\""
                    << idVar * nx * ny * sizeof(real_t) + idVar * sizeof(unsigned int)
                    << "\" />" << std::endl;
            idVar++;
        }

        outFile << "    </CellData>" << std::endl;
        outFile << "  </Piece>" << std::endl;
        outFile << "  </ImageData>" << std::endl;

        outFile << "  <AppendedData encoding=\"raw\">" << std::endl;

        // write the leading undescore
        outFile << "_";
        // then write heavy data (column major format)
        {
            unsigned int nbOfWords = nx * ny * sizeof(real_t);
            for (auto& iter : index2names) {

                // get variable id
                int iVar = iter.first;

                outFile.write((char*)&nbOfWords, sizeof(unsigned int));
                for (int j = jmin + ghostWidth; j <= jmax - ghostWidth; j++)
                    for (int i = imin + ghostWidth; i <= imax - ghostWidth; i++) {
                        real_t tmp = data_h(i, j, iVar);
                        outFile.write((char*)&tmp, sizeof(real_t));
                    }
            } // end for index2names
        }

        outFile << "  </AppendedData>" << std::endl;
        outFile << "</VTKFile>" << std::endl;

    } // end ascii/binary heavy data write

    outFile.close();

} // save_vtk_mpi - 2d

// =======================================================
// =======================================================
void save_vtk_mpi(LBMArray3dHost data_h,
    const LBMParams& params,
    const ConfigMap& configMap,
    const int2str_t& index2names,
    int iStep)
{

    // some useful parameter alias
    const int nx = params.nx;
    const int ny = params.ny;
    const int nz = params.nz;

    const int imin = params.imin;
    const int imax = params.imax;

    const int jmin = params.jmin;
    const int jmax = params.jmax;

    const int kmin = params.kmin;
    const int kmax = params.kmax;

    const int ghostWidth = params.ghostWidth;

    const real_t dx = params.dx;
    const real_t dy = params.dy;
    const real_t dz = params.dz;

    const int isize = params.isize;
    const int jsize = params.jsize;
    const int ksize = params.ksize;
    const int nbCells = isize * jsize * ksize;
    const int ijsize = isize * jsize;

    int xmin = 0, xmax = 0, ymin = 0, ymax = 0, zmin = 0, zmax = 0;
    xmin = params.myMpiPos[0] * nx;
    xmax = params.myMpiPos[0] * nx + nx;
    ymin = params.myMpiPos[1] * ny;
    ymax = params.myMpiPos[1] * ny + ny;
    zmin = params.myMpiPos[2] * nz;
    zmax = params.myMpiPos[2] * nz + nz;

    // local variables
    int i, j, k;
    std::string outputDir = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

    bool outputVtkAscii = configMap.getBool("output", "outputVtkAscii", false);

    // check scalar data type
    bool useDouble = false;

    if (sizeof(real_t) == sizeof(double)) {
        useDouble = true;
    }

    // write iStep in string stepNum
    auto stepNum = format_index(iStep, 7);

    // write MPI rank in string rankFormat
    auto rank = format_index(params.myRank, 5);

    // write pvti header file
    if (params.myRank == 0) {
        // header file : parallel vti format
        std::string headerFilename = outputDir + "/" + outputPrefix + "_time" + stepNum + ".pvti";

        write_pvti_header(headerFilename,
            outputPrefix,
            DIM3,
            params,
            index2names,
            iStep);
    }

    // concatenate file prefix + file number + suffix
    std::string filename = outputDir + "/" + outputPrefix + "_time" + stepNum + "_mpi" + rank + ".vti";

    // open file
    std::fstream outFile;
    outFile.open(filename.c_str(), std::ios_base::out);

    // write header

    // if writing raw binary data (file does not respect XML standard)
    if (outputVtkAscii)
        outFile << "<?xml version=\"1.0\"?>\n";

    if (isBigEndian()) {
        outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"BigEndian\">\n";
    } else {
        outFile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    }

    // write mesh extent
    outFile << "  <ImageData WholeExtent=\""
            << xmin << " " << xmax << " "
            << ymin << " " << ymax << " "
            << zmin << " " << zmax << "\" "
            << "Origin=\""
            << params.xmin << " " << params.ymin << " " << params.zmin << "\" "
            << "Spacing=\""
            << dx << " " << dy << " " << dz << "\">\n";
    outFile << "  <Piece Extent=\""
            << xmin << " " << xmax << " "
            << ymin << " " << ymax << " "
            << zmin << " " << zmax << " "
            << "\">\n";

    outFile << "    <PointData>\n";
    outFile << "    </PointData>\n";

    if (outputVtkAscii) {

        outFile << "    <CellData>\n";

        // write data array (ascii), remove ghost cells
        for (auto& iter : index2names) {

            // get variable id
            int iVar = iter.first;

            // get variables string name
            const std::string varName = iter.second;

            outFile << "    <DataArray type=\"";
            if (useDouble) {
                outFile << "Float64";
            } else {
                outFile << "Float32";
            }
            outFile << "\" Name=\"" << varName << "\" format=\"ascii\" >\n";

            for (int index = 0; index < nbCells; ++index) {

                // enforce the use of left layout (Ok for CUDA)
                // but for OpenMP, we will need to transpose
                k = index / ijsize;
                j = (index - k * ijsize) / isize;
                i = index - j * isize - k * ijsize;

                if (k >= kmin + ghostWidth and k <= kmax - ghostWidth and j >= jmin + ghostWidth and j <= jmax - ghostWidth and i >= imin + ghostWidth and i <= imax - ghostWidth) {
                    outFile << data_h(i, j, k, iVar) << " ";
                }
            }
            outFile << "\n    </DataArray>\n";
        } // end for iVar

        outFile << "    </CellData>\n";

        // write footer
        outFile << "  </Piece>\n";
        outFile << "  </ImageData>\n";
        outFile << "</VTKFile>\n";

    } else { // write data in binary format

        outFile << "    <CellData>" << std::endl;

        int idVar = 0;
        for (auto& iter : index2names) {

            // get variable id
            //int iVar = iter.first;

            // get variables string name
            const std::string varName = iter.second;

            if (useDouble) {
                outFile << "     <DataArray type=\"Float64\" Name=\"";
            } else {
                outFile << "     <DataArray type=\"Float32\" Name=\"";
            }
            outFile << varName
                    << "\" format=\"appended\" offset=\""
                    << idVar * nx * ny * nz * sizeof(real_t) + idVar * sizeof(unsigned int)
                    << "\" />" << std::endl;
            idVar++;

        } // end for index2names

        outFile << "    </CellData>" << std::endl;
        outFile << "  </Piece>" << std::endl;
        outFile << "  </ImageData>" << std::endl;

        outFile << "  <AppendedData encoding=\"raw\">" << std::endl;

        // write the leading undescore
        outFile << "_";

        // then write heavy data (column major format)
        {
            unsigned int nbOfWords = nx * ny * nz * sizeof(real_t);
            for (auto& iter : index2names) {

                // get variable id
                int iVar = iter.first;

                outFile.write((char*)&nbOfWords, sizeof(unsigned int));
                for (int k = kmin + ghostWidth; k <= kmax - ghostWidth; k++)
                    for (int j = jmin + ghostWidth; j <= jmax - ghostWidth; j++)
                        for (int i = imin + ghostWidth; i <= imax - ghostWidth; i++) {
                            real_t tmp = data_h(i, j, k, iVar);
                            outFile.write((char*)&tmp, sizeof(real_t));
                        }
            } // end for index2names
        }

        outFile << "  </AppendedData>" << std::endl;
        outFile << "</VTKFile>" << std::endl;

    } // end ascii/binary heavy data write

    outFile.close();

} // save_vtk_mpi - 3d

/*
 * write pvti header in a separate file.
 */
// =======================================================
// =======================================================
void write_pvti_header(std::string headerFilename,
    std::string outputPrefix,
    const DimensionType dim,
    const LBMParams& params,
    const int2str_t& index2names,
    int iStep)
{
    // file handler
    std::fstream outHeader;

    // dummy string here, when using the full VTK API, data can be compressed
    // here, no compression used
    std::string compressor("");

    // check scalar data type
    bool useDouble = false;

    if (sizeof(real_t) == sizeof(double)) {
        useDouble = true;
    }

    const int nProcs = params.nProcs;

    // write iStep in string timeFormat
    auto timeStr = format_index(iStep, 7);

    // local sub-domain sizes
    const int nx = params.nx;
    const int ny = params.ny;
    const int nz = (dim == THREE_D) ? params.nz : 0;

    // sizes of MPI Cartesian topology
    const int mx = params.mx;
    const int my = params.my;
    const int mz = (dim == THREE_D) ? params.mz : 0;

    const real_t dx = params.dx;
    const real_t dy = params.dy;
    const real_t dz = (dim == THREE_D) ? params.dz : 0.0;

    // open pvti header file
    outHeader.open(headerFilename.c_str(), std::ios_base::out);

    outHeader << "<?xml version=\"1.0\"?>" << std::endl;
    if (isBigEndian())
        outHeader << "<VTKFile type=\"PImageData\" version=\"0.1\" byte_order=\"BigEndian\"" << compressor << ">" << std::endl;
    else
        outHeader << "<VTKFile type=\"PImageData\" version=\"0.1\" byte_order=\"LittleEndian\"" << compressor << ">" << std::endl;
    outHeader << "  <PImageData WholeExtent=\"";
    outHeader << 0 << " " << mx * nx << " ";
    outHeader << 0 << " " << my * ny << " ";
    outHeader << 0 << " " << mz * nz << "\" GhostLevel=\"0\" "
              << "Origin=\""
              << params.xmin << " " << params.ymin << " " << params.zmin << "\" "
              << "Spacing=\""
              << dx << " " << dy << " " << dz << "\">"
              << std::endl;
    outHeader << "    <PCellData Scalars=\"Scalars_\">" << std::endl;

    for (auto& iter : index2names) {

        // get variables string name
        const std::string varName = iter.second;

        if (useDouble)
            outHeader << "      <PDataArray type=\"Float64\" Name=\"" << varName << "\"/>" << std::endl;
        else
            outHeader << "      <PDataArray type=\"Float32\" Name=\"" << varName << "\"/>" << std::endl;
    }
    outHeader << "    </PCellData>" << std::endl;

    // one piece per MPI process
    if (dim == TWO_D) {

        for (int iPiece = 0; iPiece < nProcs; ++iPiece) {
            auto pieceStr = format_index(iPiece, 5);
            std::string pieceFilename = outputPrefix + "_time" + timeStr + "_mpi" + pieceStr + ".vti";
            // get MPI coords corresponding to MPI rank iPiece
            int coords[2];
            params.communicator->getCoords(iPiece, 2, coords);
            outHeader << "    <Piece Extent=\"";

            // pieces in first line of column are different (due to the special
            // pvti file format with overlapping by 1 cell)
            if (coords[0] == 0)
                outHeader << 0 << " " << nx << " ";
            else
                outHeader << coords[0] * nx << " " << coords[0] * nx + nx << " ";
            if (coords[1] == 0)
                outHeader << 0 << " " << ny << " ";
            else
                outHeader << coords[1] * ny << " " << coords[1] * ny + ny << " ";
            outHeader << 0 << " " << 1 << "\" Source=\"";
            outHeader << pieceFilename << "\"/>" << std::endl;
        }

    } else { // THREE_D

        for (int iPiece = 0; iPiece < nProcs; ++iPiece) {
            auto pieceStr = format_index(iPiece, 5);
            std::string pieceFilename = outputPrefix + "_time" + timeStr + "_mpi" + pieceStr + ".vti";
            // get MPI coords corresponding to MPI rank iPiece
            int coords[3];
            params.communicator->getCoords(iPiece, 3, coords);
            outHeader << " <Piece Extent=\"";

            if (coords[0] == 0)
                outHeader << 0 << " " << nx << " ";
            else
                outHeader << coords[0] * nx << " " << coords[0] * nx + nx << " ";

            if (coords[1] == 0)
                outHeader << 0 << " " << ny << " ";
            else
                outHeader << coords[1] * ny << " " << coords[1] * ny + ny << " ";

            if (coords[2] == 0)
                outHeader << 0 << " " << nz << " ";
            else
                outHeader << coords[2] * nz << " " << coords[2] * nz + nz << " ";

            outHeader << "\" Source=\"";
            outHeader << pieceFilename << "\"/>" << std::endl;
        }
    }
    outHeader << "  </PImageData>" << std::endl;
    outHeader << "</VTKFile>" << std::endl;

    // close header file
    outHeader.close();

    // end writing pvti header

} // write_pvti_header

#endif // USE_MPI

} // namespace lbm_saclay
