#ifndef VECTOR_TYPES_H_
#define VECTOR_TYPES_H_

using IVect2 = typename Kokkos::Array<int, 2>;
using IVect3 = typename Kokkos::Array<int, 3>;
template <int dim>
using IVect = typename std::conditional<dim == 2, IVect2, IVect3>::type;

using RVect2 = typename Kokkos::Array<real_t, 2>;
using RVect3 = typename Kokkos::Array<real_t, 3>;
template <int dim>
using RVect = typename std::conditional<dim == 2, RVect2, RVect3>::type;



// vector operations

// IVect add
KOKKOS_INLINE_FUNCTION
IVect2 IVectAdd(const IVect2& V1, const IVect2& V2)
{
    return IVect2({V1[IX] + V2[IX] , V1[IY] + V2[IY]});
}
KOKKOS_INLINE_FUNCTION
IVect3 IVectAdd(const IVect3& V1, const IVect3& V2)
{
	return IVect3({V1[IX] + V2[IX] , V1[IY] + V2[IY] , V1[IZ] + V2[IZ]});
}


// RVect add
KOKKOS_INLINE_FUNCTION
RVect2 RVectAdd(const RVect2& V1, const RVect2& V2)
{
    return RVect2({V1[IX] + V2[IX] , V1[IY] + V2[IY]});
}
KOKKOS_INLINE_FUNCTION
RVect3 RVectAdd(const RVect3& V1, const RVect3& V2)
{
	return RVect3({V1[IX] + V2[IX] , V1[IY] + V2[IY] , V1[IZ] + V2[IZ]});
}

template<int dim>
RVect<dim> operator+(RVect<dim> const& V1, RVect<dim> const& V2)
{
    return RVectAdd(V1,V2);
}
//~ template<int dim>
//~ RVect<dim> operator+(RVect<dim>* V1, RVect<dim>* V2)
//~ {
    //~ return &RVectAdd(*V1,*V2);
//~ }

// RVect scalar product
KOKKOS_INLINE_FUNCTION
real_t scalar_product(const RVect2& V1, const RVect2& V2)
{
    return V1[IX] * V2[IX] + V1[IY] * V2[IY];
}
KOKKOS_INLINE_FUNCTION
real_t scalar_product(const RVect3& V1, const RVect3& V2)
{
    return V1[IX] * V2[IX] + V1[IY] * V2[IY] + V1[IZ] * V2[IZ];
}

#endif // KOKKOS_SHARED_H_
