#ifndef CC_LABELLING_H_
#define CC_LABELLING_H_

#include "LBM_Base_Functor.h"




template<typename Scheme>
struct FunctorCustomUpdateCClabels
{
    Scheme scheme;
    
    KOKKOS_INLINE_FUNCTION
    FunctorCustomUpdateCClabels(const Scheme& scheme):
        scheme(scheme) {};

    KOKKOS_INLINE_FUNCTION
    void operator ()(const IVect2& IJK) const
    {


        int isize=scheme.params.isize;
        int jsize=scheme.params.jsize;
        //~ int gw=Base::params.ghostWidth;
        if (IJK[IX] >0 && IJK[IX]<isize-1 && IJK[IY]>0 && IJK[IY]<jsize-1)
        {
            real_t eps=scheme.get_cc_label(IJK);


            // cartesian mask
            real_t l1=scheme.get_cc_label(IVectAdd(IJK, { 1, 0}));
            real_t l2=scheme.get_cc_label(IVectAdd(IJK, {-1, 0}));
            real_t l3=scheme.get_cc_label(IVectAdd(IJK, { 0, 1}));
            real_t l4=scheme.get_cc_label(IVectAdd(IJK, { 0,-1}));






            // update label
            if (eps>0)
            {

                // implement warp0 optimization
                // principle: look at source of current CC to see if it was overcome by another source
                real_t ls=0;



                IVect2 sIJK = scheme.CCindexToIVect2(eps);
                // source can belong to another MPI process,
                //for now, ignore the problem if it happens
                //will think of how to handle this later because right now, this is terrible for performance
                bool sourceIsInBounds= scheme.is_in_bounds(scheme.tagPHI,sIJK);

                if (sourceIsInBounds)
                {
                    ls=scheme.get_cc_label(sIJK);
                }

                eps=FMAX(FMAX(eps,ls),FMAX(FMAX(l1,l2),FMAX(l3,l4)));


                scheme.set_cc_label(IJK, eps);
            }
        }


    };


    KOKKOS_INLINE_FUNCTION
    void operator ()(const IVect3& IJK) const
    {
        int isize=scheme.params.isize;
        int jsize=scheme.params.jsize;
        int ksize=scheme.params.ksize;
        //~ int gw=Base::params.ghostWidth;
        if (IJK[IX] >0 && IJK[IX]<isize-1 && IJK[IY]>0 && IJK[IY]<jsize-1 && IJK[IZ]>0 && IJK[IZ]<ksize-1)
        {
            real_t eps=scheme.get_cc_label(IJK);


            // cartesian mask
            real_t l1=scheme.get_cc_label(IVectAdd(IJK, { 1, 0, 0}));
            real_t l2=scheme.get_cc_label(IVectAdd(IJK, {-1, 0, 0}));
            real_t l3=scheme.get_cc_label(IVectAdd(IJK, { 0, 1, 0}));
            real_t l4=scheme.get_cc_label(IVectAdd(IJK, { 0,-1, 0}));
            real_t l5=scheme.get_cc_label(IVectAdd(IJK, { 0, 0, 1}));
            real_t l6=scheme.get_cc_label(IVectAdd(IJK, { 0, 0,-1}));






            // update label
            if (eps>0)
            {

                // implement warp0 optimization
                // principle: look at source of current CC to see if it was overcome by another source
                real_t ls=0;



                IVect3 sIJK = scheme.CCindexToIVect3(eps);

                bool sourceIsInBounds= scheme.is_in_bounds(scheme.tagPHI,sIJK);

                if (sourceIsInBounds)
                {
                    ls=scheme.get_cc_label(sIJK);
                }



                eps=FMAX(FMAX(eps,ls),FMAX(FMAX(FMAX(l1,l2),FMAX(l3,l4)),FMAX(l5,l6)));


                scheme.set_cc_label(IJK, eps);
            }
        }


    };

};// end functor update labels

template<typename Scheme>
struct FunctorMakeCCidList
{
    Scheme scheme;
    
    KOKKOS_INLINE_FUNCTION
    FunctorMakeCCidList(const Scheme& scheme):
        scheme(scheme) {};

    KOKKOS_INLINE_FUNCTION
    void operator ()(const IVect<Scheme::tp_dim>& IJK) const
    {

        bool isInBounds= scheme.is_in_bounds(scheme.tagPHI,IJK);

        if (isInBounds)
        {
            real_t cc = scheme.get_cc_label(IJK);
            int cci=int(cc);

            scheme.map_CCid.insert(cci);
        }

    };

};

template<typename Scheme>
  struct FunctorCustomRescaleCClabels
    {
        Scheme scheme;
        
        KOKKOS_INLINE_FUNCTION
        FunctorCustomRescaleCClabels(const Scheme& scheme):
            scheme(scheme) {};

        KOKKOS_INLINE_FUNCTION
        void operator ()(const IVect<Scheme::tp_dim>& IJK) const
        {

            real_t cc = scheme.get_cc_label(IJK);

            if (cc>0.5)
            {


                uint32_t id1=scheme.map_CCid_to_rank.find(int(cc));
                if( scheme.map_CCid_to_rank.valid_at(id1) )
                {
                    real_t rank=scheme.map_CCid_to_rank.value_at(id1);

                    scheme.set_cc_label(IJK, (1.0*rank));
                }

            }
        };

    };
    
template<typename Scheme>
struct FunctorSumCC
{
    Scheme scheme;
    
    KOKKOS_INLINE_FUNCTION
    FunctorSumCC(const Scheme& scheme):
        scheme(scheme) {};

    KOKKOS_INLINE_FUNCTION
    real_t operator ()(const IVect<Scheme::tp_dim>& IJK) const
    {
        real_t dx=scheme.params.dx;
        real_t dv=Scheme::tp_dim==2 ? dx*dx : dx*dx*dx;
        real_t CC = dv*scheme.get_cc_label(IJK);

        return CC;
    };

}; // end functor CC


#endif
