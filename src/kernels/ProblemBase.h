#ifndef PROBLEM_BASE_H_
#define PROBLEM_BASE_H_



#include <stdVectorUtils.h>

#include "LBMParams.h"
#include "LBM_Base_Functor.h"
#include "ConnectedComponentsLabellingFunctors.h"

template <int dim, int npop>
struct ProblemBase
{

    static constexpr int t_dim=dim;
    static constexpr int t_npop=npop;

    using FArray = typename LBMBaseFunctor<dim, npop>::FArray;
    using FState = typename Kokkos::Array<real_t, npop>;
    using LBMArray = typename std::conditional<dim == 2, LBMArray2d, LBMArray3d>::type;
    using LBMArrayHost = typename LBMArray::HostMirror;

    ConfigMap& configMap;
    LBMParams& params;
    //! names of variables to save

    LBMArray Mdata; /*!< LBM fields (density, vx,vy) array */
    LBMArrayHost Mdata_h; /*!< data mirror on host memory space */

    int nbEqs;
    int nbVar;

    //! io writer
    std::shared_ptr<lbm_saclay::IO_ReadWrite> m_io_reader_writer;
    PeriodicManager<dim, npop> bcPerMgr;
    PeriodicManager<dim, 1> bcPerMgr_cc;

    int2str_t index2names_allvars;
    int2str_t index2names_vars_to_write;

    // constructor for 2 equation (ie 2 lbm distribution of type FArray)
    ProblemBase(ConfigMap& configMap, LBMParams& params, int2str_t index2names_allvars)
        : configMap(configMap)
        , params(params)
        , bcPerMgr(params)
        , bcPerMgr_cc(params)
        , index2names_allvars(index2names_allvars)
        , index2names_vars_to_write(lbm_saclay::build_var_to_write_map(params, configMap, index2names_allvars))
    {
        // list of pair (index,names) of macroscopic variables to write int2str_t index2names;

        m_io_reader_writer = std::make_shared<lbm_saclay::IO_ReadWrite>(params, configMap, index2names_allvars);

        if (params.myRank == 0)
        {
            if (params.collisionType1 == BGK)
            {
                std::cout << "using BGK collision for equation 1" << std::endl;
            }
            else if (params.collisionType1 == TRT)
            {
                std::cout << "using TRT collision for equation 1" << std::endl;
            }
            else if (params.collisionType1 == MRT)
            {
                std::cout << "using MRT collision for equation 1" << std::endl;
            }

            if (params.collisionType2 == BGK)
            {
                std::cout << "using BGK collision for equation 2" << std::endl;
            }
            else if (params.collisionType2 == TRT)
            {
                std::cout << "using TRT collision for equation 2" << std::endl;
            }
            else if (params.collisionType2 == MRT)
            {
                std::cout << "using MRT collision for equation 2" << std::endl;
            }

            if (params.collisionType3 == BGK)
            {
                std::cout << "using BGK collision for equation 3" << std::endl;
            }
            else if (params.collisionType3 == TRT)
            {
                std::cout << "using TRT collision for equation 3" << std::endl;
            }
            else if (params.collisionType3 == MRT)
            {
                std::cout << "using MRT collision for equation 3" << std::endl;
            }

            if (params.collisionType4 == BGK)
            {
                std::cout << "using BGK collision for equation 4" << std::endl;
            }
            else if (params.collisionType4 == TRT)
            {
                std::cout << "using TRT collision for equation 4" << std::endl;
            }
            else if (params.collisionType4 == MRT)
            {
                std::cout << "using MRT collision for equation 4" << std::endl;
            }
        }
    };

    virtual ~ProblemBase() {};

    virtual void init()
    {
        std::cout << "this shouldn't be used, dummy before I replace some things" << std::endl;
    };
    virtual void init_m()
    {
        std::cout << "using virtual base init_m function" << std::endl;
    };
    virtual void init_f()
    {
        std::cout << "using virtual base init_f function" << std::endl;
    };

    virtual void update_f()
    {
        std::cout << "using virtual base update_f function" << std::endl;
    };
    virtual void update_m()
    {
        std::cout << "using virtual base update_m function" << std::endl;
    };
    virtual void update_dt()
    {
        std::cout << "using virtual base update_dt function" << std::endl;
    };
    virtual void update_total_time(real_t m_t)
    {
        std::cout << "using virtual base update_total_time function" << std::endl;
    };

    virtual void make_boundaries()
    {
        std::cout << "using virtual base make_boundaries function" << std::endl;
    };
    virtual void make_periodic_boundaries()
    {

    }

//! hook functions to add actions at certain specific points during execution of main loop
    virtual void hook_before_output(int step)
    {

    }
    virtual void hook_before_final_output()
    {

    }



    virtual void finalize()
    {
        std::cout << "using virtual base finalize function" << std::endl;
    };

    virtual real_t get_dt() = 0;

    virtual void loadDataCSV(LBMArray2d ldata) {};
    virtual void loadDataCSV(LBMArray3d ldata) {};

    int get_nbEqs()
    {
        return nbEqs;
    };
    int get_nbVar()
    {
        return nbVar;
    };
    int2str_t get_var_to_write()
    {
        return index2names_vars_to_write;
    };

    void writeData(int m_step, real_t m_t, int m_times_saved)
    {
        m_io_reader_writer->save_data(Mdata, Mdata_h, m_step, m_t, m_times_saved, index2names_vars_to_write);
    };
    void writeDataFull(int m_step, real_t m_t, int m_times_saved)
    {
        m_io_reader_writer->save_data(Mdata, Mdata_h, m_step, m_t, m_times_saved, index2names_allvars);
    };
    void loadDataFull(int& m_step, real_t& m_t, int& m_times_saved)
    {
        m_io_reader_writer->load_data(Mdata, Mdata_h, m_step, m_t, m_times_saved);
    };

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    template<typename Scheme, typename FunctorInitlabels>
    void compute_connected_components(Scheme& scheme)
    {

        using FunctorSumLabels=FunctorSumCC<Scheme>;
        using FunctorUpdateLabels=FunctorCustomUpdateCClabels<Scheme>;


        bool output_CC_info = configMap.getBool("params", "output_CC_info", false);

        real_t sum_CC=0;
        real_t sum_CC_prev=0;
        //~ std::cout<<"ccc: init" << std::endl;

        CustomFunctor<dim, npop, Scheme, FunctorInitlabels>::apply(params, scheme);
        bcPerMgr_cc.make_boundaries(scheme.CCdata, BOUNDARY_EQUATION_1);
        //~ std::cout<<"ccc: sum CC" << std::endl;



        ReducerSum<dim, npop, Scheme, FunctorSumLabels>::apply(params, scheme, sum_CC);
        sum_CC_prev = sum_CC-1;

        int niter=0;
        //!loop until total value stop decreasing
        //~ std::cout<<"ccc: start loop" << std::endl;
        while (sum_CC_prev<sum_CC)
        {
            niter++;


            //~ if (params.myRank==0) printf("Rank %d, CCC iter %d: prev_sum = %g, curr sum = %g \n", params.myRank, niter, (sum_CC_prev), (sum_CC));
            sum_CC_prev = sum_CC;



            CustomFunctor<dim, npop, Scheme, FunctorUpdateLabels>::apply(params, scheme);


            bcPerMgr_cc.make_boundaries(scheme.CCdata, BOUNDARY_EQUATION_1);

            //~ printf("Rank %d, CCC iter %d: reduce \n", params.myRank, niter);

            ReducerSum<dim, npop, Scheme, FunctorSumLabels>::apply(params, scheme, sum_CC);


            //~ printf("Rank %d, CCC iter %d: new_sum = %d \n", params.myRank, niter, int(sum_CC));
            //~ printf("Rank %d, CCC iter %d: end \n", params.myRank, niter);
        }

        if (output_CC_info and params.myRank==0)
        {

            std::cout<<"MPI rank : " << params.myRank<< " | niter: "<<niter<<std::endl;

        }


        m_rescale_cc_ids(scheme);
        //~ std::cout<<"ccc: end" << std::endl;
    }



//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// function to rescale connected components labels in output, so that they are numeroted in order from 1, 2 ...
    template<typename Scheme>
    void m_rescale_cc_ids(Scheme& scheme)
    {

        using FunctorListLabels=FunctorMakeCCidList<Scheme>;
        


        //~ printf("Rank %d, rescale, start \n", params.myRank);

        //! clear map to ranks on device
        scheme.map_CCid.clear();
        scheme.map_CCid_to_rank.clear();

        //! fill list of cc labels (per mpi process)
        CustomFunctor<dim, npop, Scheme, FunctorListLabels>::apply(params, scheme);
#ifdef USE_MPI
        params.communicator->synchronize();
#endif

        uset_h map_CCid_h(KOKKOS_MAP_DEFAULT_CAPACITY);
        umap_h map_CCid_to_rank_h(KOKKOS_MAP_DEFAULT_CAPACITY);
        Kokkos::deep_copy(map_CCid_h, scheme.map_CCid);

        const uint32_t n=map_CCid_h.capacity();
        //~ std::cout <<"n="<< n << std::endl;
        std::vector<int> arr = {};

        //! get the labels from the kokkos map
        for (uint32_t i=0; i<n; i++)
        {

            if( map_CCid_h.valid_at(i) )
            {
                auto key   = map_CCid_h.key_at(i);
                //std::cout << i <<"  "<< key << std::endl;
                arr.push_back(key);
            }

        }
        //~ std::cout <<arr.size() << std::endl;
        //~ printf("Rank %d, rescale, found %d labels on this MPI process\n", params.myRank, int(arr.size()));


#ifdef USE_MPI
        arr.resize(n,0);
        //~ printf("Rank %d, rescale, resized array to size  %d\n", params.myRank, int(arr.size()));params.communicator->synchronize();
        //! gather the labels
        std::vector<int> recvbuf(params.nProcs*n);
        //~ recvbuf.reserve(params.nProcs*n);
        //~ printf("Rank %d, rescale, created buffer of size %d and capacity %d\n", params.myRank, int(recvbuf.size()),  int(recvbuf.capacity()));params.communicator->synchronize();
        params.communicator->allGather(arr.data(), n, hydroSimu::MpiComm::INT, recvbuf.data(), n, hydroSimu::MpiComm::INT);
        //~ printf("Rank %d, rescale, gathered in buffer of size %d and capacity %d\n", params.myRank, int(recvbuf.size()),  int(recvbuf.capacity()));params.communicator->synchronize();
        //! build list of unique labels
        arr.clear();
        stdVectorUtils::list_uniques(recvbuf, arr);
        //~ printf("Rank %d, rescale, listed %d unique labels \n", params.myRank, int(arr.size()));params.communicator->synchronize();
#endif
        bool output_CC_info = configMap.getBool("params", "output_CC_info", false);
        if (output_CC_info && params.myRank==0)
        {
            std::cout<<"MPI rank : " << params.myRank<<" | countcc    : "<<arr.size()<<std::endl;

        }
        //~ std::cout <<"end scan" << std::endl;
        std::sort(arr.begin(), arr.end(), std::less<int>());

        //~ printf("Rank %d, rescale, sorted array \n", params.myRank);


        for (uint32_t rank=0; rank<arr.size(); rank++)
        {
            int key   = arr[rank];


            map_CCid_to_rank_h.insert(key, (rank));
            //printf("Rank %d, rescale, CC no %d is rank %d \n", params.myRank, key, rank);


        }



        Kokkos::deep_copy(scheme.map_CCid_to_rank, map_CCid_to_rank_h);
        //~ std::cout <<"end ranks" << std::endl;
        CustomFunctor<dim, npop, Scheme, FunctorCustomRescaleCClabels<Scheme>>::apply(params, scheme);
        //~ printf("Rank %d, rescale, end \n", params.myRank);
    }

#ifdef USE_MPI
    void mpi_sum_map(umap& map)
    {

        umap_h map_h(KOKKOS_MAP_DEFAULT_CAPACITY);
        Kokkos::deep_copy(map_h, map);

        const uint32_t n=map_h.capacity();

        std::vector<real_t> arr(n);

        //! get the labels from the kokkos map
        for (uint32_t i=0; i<n; i++)
        {

            if( map_h.valid_at(i) )
            {
                auto key   = map_h.key_at(i);
                //~ std::cout << i <<"  "<< key << std::endl;
                arr[key]=map_h.value_at(i);
            }

        }

        //~ char buffer[50];
        //~ std::string str1;
        //~ sprintf(buffer, "arr | MPI %d : ", params.myRank);
        //~ str1+=std::string(buffer);
        //~ for (uint32_t i=0;i<arr.size();i++)
        //~ {
        //~ real_t val   = arr[i];
        //~ if (val>0)
        //~ {
        //~ sprintf(buffer, "[%u | %g ]",i, val);
        //~ str1+=std::string(buffer);
        //~ }
        //~ }
        //~ std::cout<<str1 <<std::endl;

        std::vector<real_t> arr_reduced(n);
        //! MPI allreduce for array
        params.communicator->allReduce(arr.data(), arr_reduced.data(), n, params.data_type,  hydroSimu::MpiComm::SUM);


        //~ std::string str2;
        //~ sprintf(buffer, "red | MPI %d : ", params.myRank);
        //~ str2+=std::string(buffer);
        //~ for (uint32_t i=0;i<arr_reduced.size();i++)
        //~ {
        //~ real_t val   = arr_reduced[i];
        //~ if (val>0)
        //~ {
        //~ sprintf(buffer, "[%u | %g ]",i, val);
        //~ str2+=std::string(buffer);
        //~ }
        //~ }
        //~ std::cout<<str2 <<std::endl;



        for(uint32_t i=0; i<n; i++)
        {
            map_h.insert(i, arr_reduced[i]);

        }

        Kokkos::deep_copy(map,map_h);

    }
#endif


}; // end class ProblemBase

#endif // LBM_SCHEME_BASE_H_
