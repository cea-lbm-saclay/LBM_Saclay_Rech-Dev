#ifndef IO_CSV_H_
#define IO_CSV_H_

#include <fstream>
#include <vector>
#include <filesystem> // for std::filesystem

#include "../LBMParams.h"
#include "../kokkos_shared.h"







struct csvReader
{

    LBMParams params;

    csvReader(const LBMParams& params)
        : params(params) {};


    std::vector<std::string> splitstring(std::string s, std::string delimiter)
    {
        size_t pos_start = 0, pos_end, delim_len = delimiter.length();
        std::string token;
        std::vector<std::string> res;

        while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos)
        {
            token = s.substr (pos_start, pos_end - pos_start);
            pos_start = pos_end + delim_len;
            res.push_back (token);
        }

        res.push_back (s.substr (pos_start));
        return res;
    }


    void loadFieldFromCSV(LBMArray2d ldata,
                          int ifield,
                          std::string filename)
    {

        //~ std::cout << "csvReader: open file: "<< filename.c_str() << std::endl;
        //! if using mpi, append process position to filename
#ifdef USE_MPI

        std::vector<std::string> splitted=splitstring(filename, ".");

        if (splitted.size() !=2)
        {
            std::cout <<"MPI process "<<params.myRank<< ", file has incorrect format: " << filename << std::endl;
            exit(1);
        }
        char pos_indicator[50];
        sprintf(pos_indicator, "%d%d",params.myMpiPos[IX],params.myMpiPos[IY]);
        filename=splitted[0].append(pos_indicator).append(".").append(splitted[1]);


#endif
        if (not(std::filesystem::exists(filename)))
        {
            printf("MPI process %d, Trying to load data from non-existent file %s\n", params.myRank, filename.c_str());
            exit(1);
        }

        printf("MPI process %d, reading from file %s", params.myRank, filename.c_str());
        std::cout<<std::endl;

        std::ifstream myFile(filename);

        int sizeRatio = params.sizeRatio;
        real_t normalizeInputPhi = params.normalizeInputPhi;
        //~ std::cout << "csvReader: sizeRatio         = "<< sizeRatio << std::endl;
        //~ std::cout << "csvReader: normalizeInputPhi = "<< normalizeInputPhi << std::endl;


        const int gw = params.ghostWidth;
        const int nx = params.nx;
        const int ny = params.ny;

        // temporary host array to read data before copying into device array
        ArrayScalar2d readData_device = ArrayScalar2d("readData", nx, ny);
        ArrayScalar2dHost readData = Kokkos::create_mirror_view(readData_device);

        // Helper vars
        std::string line, word;

        Kokkos::Array<int, 2> indval;
        real_t value = 0.0;
        
        int current_line=0;
        // Read data, line by line
        while (std::getline(myFile, line))
        {
            current_line++;
            // Create a stringstream of the current line
            std::stringstream ss(line);

            // Keep track of the current column index
            int colIdx = 0;

            // Extract each word between comma on the line
            while (std::getline(ss, word, ','))
            {

                if (colIdx < 2)
                {
                    // Add the current integer to the 'colIdx' column's values vector
                    indval[colIdx] = std::stoi(word);
                }
                else
                {
                    value = std::stod(word);
                }
                colIdx++;
            }

            for (int i = 0; i < sizeRatio; i++)
            {
                for (int j = 0; j < sizeRatio; j++)
                {
                    const unsigned long int ipos = sizeRatio * indval[0] + i;
                    const unsigned long int jpos = sizeRatio * indval[1] + j;
                    if ( ipos<readData.extent(0) and jpos< readData.extent(1)) // no need to check >=0 as ipos and jpos are unsigned
                    {
                        readData(ipos,jpos) = (value / normalizeInputPhi);
                    } else 
                        {
                            char err_msg[150];
                            sprintf(err_msg,"invalid indices encountered (%d,%d) while reading csv input at line %d", indval[0], indval[1], current_line);
                            throw std::runtime_error(err_msg);
                        }
                }
            }
        }

        // Close file
        myFile.close();

        // take a subview in LBM data where we want to copy read data into
        auto ldata_phi = Kokkos::subview(ldata, std::make_pair(gw, gw + nx),
                                         std::make_pair(gw, gw + ny), int(ifield));

        // transfer read data to device memory
        //~ std::cout << "csvReader: send data to device" << std::endl;
        Kokkos::deep_copy(readData_device, readData);
        // copy read data into main data array
        //~ std::cout << "csvReader: copy data to lbm_array" << std::endl;
        Kokkos::deep_copy(ldata_phi, readData_device);
        //~ std::cout << "csvReader: finished reading csv" << std::endl;

    } // LBMRun<dim,npop>::loadImageData - 2d;

    void loadFieldFromCSV(LBMArray3d ldata,
                          int ifield,
                          std::string filename)
    {


        //! if using mpi, append process position to filename
#ifdef USE_MPI

        std::vector<std::string> splitted=splitstring(filename, ".");

        if (splitted.size() !=2)
        {
            std::cout <<"MPI process "<<params.myRank<< ", file has incorrect format: " << filename << std::endl;
            exit(1);
        }
        char pos_indicator[50];
        sprintf(pos_indicator, "%d%d%d",params.myMpiPos[IX],params.myMpiPos[IY],params.myMpiPos[IZ]);
        filename=splitted[0].append(pos_indicator).append(".").append(splitted[1]);


#endif
        if (not(std::filesystem::exists(filename)))
        {
            printf("MPI process %d, Trying to load data from non-existent file %s \n", params.myRank, filename.c_str());
            exit(1);
        }

        printf("MPI process %d, reading from file %s", params.myRank, filename.c_str());
        std::cout<<std::endl;

        const int gw = params.ghostWidth;

        std::ifstream myFile(filename);

        int sizeRatio = params.sizeRatio;
        real_t normalizeInputPhi = params.normalizeInputPhi;
        printf("sizeRatio  :  %d\n", sizeRatio);
        printf("normalizeInputPhi  :  %g\n", normalizeInputPhi);

        // temporary host array to read data before copying into device array
        const int nx = params.nx;
        const int ny = params.ny;
        const int nz = params.nz;
        ArrayScalar3d readData_device = ArrayScalar3d("readData", nx, ny, nz);
        ArrayScalar3dHost readData = Kokkos::create_mirror_view(readData_device);

        // Helper vars
        std::string line, word;

        Kokkos::Array<int, 3> indval;
        real_t value = 0.0;
        int current_line=0;
        // Read data, line by line
        while (std::getline(myFile, line))
        {
            current_line++;
            // Create a stringstream of the current line
            std::stringstream ss(line);

            // Keep track of the current column index
            int colIdx = 0;

            // Extract each word between comma on the line
            while (std::getline(ss, word, ','))
            {
                if (colIdx < 3)
                {
                    // Add the current integer to the 'colIdx' column's values vector
                    indval[colIdx] = std::stoi(word);
                }
                else
                {
                    value = std::stod(word);
                }
                colIdx++;
            }

            for (int i = 0; i < sizeRatio; i++)
            {
                for (int j = 0; j < sizeRatio; j++)
                {
                    for (int k = 0; k < sizeRatio; k++)
                    {

                        const unsigned long int ipos = sizeRatio * indval[0] + i;
                        const unsigned long int jpos = sizeRatio * indval[1] + j;
                        const unsigned long int kpos = sizeRatio * indval[2] + k;
                        if (ipos <readData.extent(0) and 
                            jpos < readData.extent(1) and 
                            kpos < readData.extent(2)) // no need to check >=0 as ipos and jpos are unsigned
                        {
                            readData(ipos, jpos, kpos) = (value / normalizeInputPhi);
                        } else 
                        {
                            char err_msg[150];
                            sprintf(err_msg,"invalid indices encountered (%d,%d,%d) while reading csv input at line %d", indval[0], indval[1], indval[2], current_line);
                            throw std::runtime_error(err_msg);
                        }
                    }
                }
            }
        }

        // Close file
        myFile.close();

        // take a subview in LBM data where we want to copy read data into
        auto ldata_phi = Kokkos::subview(ldata, std::make_pair(gw, gw + nx),
                                         std::make_pair(gw, gw + ny),
                                         std::make_pair(gw, gw + nz), int(ifield));

        // copy read data into LBM device data
        Kokkos::deep_copy(readData_device, readData);
        Kokkos::deep_copy(ldata_phi, readData_device);
    }
};

struct csvWriter
{

    LBMParams params;

    csvWriter(const LBMParams& params)
        : params(params) {};

    void writeFieldAsCSV(LBMArray2d ldata,
                         int ifield,
                         std::string filename)
    {
        std::ofstream myFile;

        myFile.open(filename);
        myFile.precision(5);
        const int nx = params.nx;
        const int ny = params.ny;
        const int gw = params.ghostWidth;

        // take a subview in LBM data of the data we want to write
        auto ldata_phi = Kokkos::subview(ldata, std::make_pair(gw, gw + nx),
                                         std::make_pair(gw, gw + ny),
                                         int(ifield));

        auto host_data = Kokkos::create_mirror_view(ldata_phi);
        // copy data to host
        Kokkos::deep_copy(host_data, ldata_phi);

        std::string sep = ",";
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                myFile << i << sep << j << sep << std::fixed << ldata_phi(i, j) << std::endl;
            }
        }

        // Close file
        myFile.close();
    }

    void writeFieldAsCSV(LBMArray3d ldata,
                         int ifield,
                         std::string filename)
    {

        std::ofstream myFile;

        myFile.open(filename);

        const int nx = params.nx;
        const int ny = params.ny;
        const int nz = params.nz;
        const int gw = params.ghostWidth;

        // take a subview in LBM data of the data we want to write
        auto ldata_phi = Kokkos::subview(ldata, std::make_pair(gw, gw + nx),
                                         std::make_pair(gw, gw + ny),
                                         std::make_pair(gw, gw + nz), int(ifield));

        auto host_data = Kokkos::create_mirror_view(ldata_phi);
        // copy data to host
        Kokkos::deep_copy(host_data, ldata_phi);

        std::string sep = ",";
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                for (int k = 0; k < nz; k++)
                {
                    myFile << i << sep << j << sep << k << sep << std::fixed << ldata_phi(i, j, k) << std::endl;
                }
            }
        }

        // Close file
        myFile.close();
    }
};

#endif // IO_CSV_H_
