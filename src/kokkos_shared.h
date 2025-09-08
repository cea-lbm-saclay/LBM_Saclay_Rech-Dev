#ifndef KOKKOS_SHARED_H_
#define KOKKOS_SHARED_H_

#include <Kokkos_Core.hpp>
//~ #include <Kokkos_Parallel.hpp>
//~ #include <Kokkos_View.hpp>
#include "real_type.h"
#include <Kokkos_Random.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include <unordered_map>
// TODO: put the following utilities elsewhere...
// make the compiler ignore an unused variable
#ifndef UNUSED
#define UNUSED(x) ((void)(x))
#endif

#ifndef MAYBE_UNUSED_CPP17
#define MAYBE_UNUSED_CPP17 [[maybe_unused]]
#endif

// make the compiler ignore an unused function
#ifdef __GNUC__
#define UNUSED_FUNCTION __attribute__((unused))
#else
#define UNUSED_FUNCTION
#endif

/**
 * enum use in io routines.
 */
enum KokkosLayout {
    KOKKOS_LAYOUT_LEFT,
    KOKKOS_LAYOUT_RIGHT
};


enum KokkosDefaultCapacity {
    KOKKOS_MAP_DEFAULT_CAPACITY=1000,
};

//~ using LBMState    = typename Kokkos::Array<real_t,COMPONENT_SIZE>;

using Device = Kokkos::DefaultExecutionSpace;
using Host = Kokkos::DefaultHostExecutionSpace;

//! a 2D scalar field array
using ArrayScalar2d = Kokkos::View<real_t**, Device>;
using ArrayScalar2dHost = ArrayScalar2d::HostMirror;

//! a 3D scalar field array
using ArrayScalar3d = Kokkos::View<real_t***, Device>;
using ArrayScalar3dHost = ArrayScalar3d::HostMirror;

using LBMArray2d = Kokkos::View<real_t***, Device>;
using LBMArray2dHost = LBMArray2d::HostMirror;
using LBMArrayConst2d = Kokkos::View<const real_t***, Device>;
using FArray2d = LBMArray2d;
using FArray2dHost = FArray2d::HostMirror; // added
using FArrayConst2d = LBMArrayConst2d;

using LBMArray3d = Kokkos::View<real_t****, Device>;
using LBMArray3dHost = LBMArray3d::HostMirror;
using LBMArrayConst3d = Kokkos::View<const real_t****, Device>;
using FArray3d = LBMArray3d;
using FArray3dHost = FArray3d::HostMirror; // added
using FArrayConst3d = LBMArrayConst3d;

//! maps and sets with kokkos
typedef Kokkos::UnorderedMap<int, void> uset;
typedef Kokkos::UnorderedMap<int, void, Kokkos::DefaultHostExecutionSpace> uset_h;

typedef Kokkos::UnorderedMap<int, real_t> umap;
typedef Kokkos::UnorderedMap<int, real_t, Kokkos::DefaultHostExecutionSpace> umap_h;


using ArrayScalar1d = Kokkos::View<real_t*, Device>;
using UnorderedMapInsertOpTypes = Kokkos::UnorderedMapInsertOpTypes<ArrayScalar1d, uint32_t>;
typedef  UnorderedMapInsertOpTypes::AtomicAdd AtomicAdd;
    
using ArrayScalar1dHost = Kokkos::View<real_t*, Host>;
using UnorderedMapInsertOpTypesHost = Kokkos::UnorderedMapInsertOpTypes<ArrayScalar1dHost, uint32_t>;
typedef  UnorderedMapInsertOpTypesHost::AtomicAdd AtomicAddHost;
    
using RANDOM_POOL = Kokkos::Random_XorShift1024_Pool<>;

/**
 * a dummy swap device routine.
 * Avoid using name swap (already existing in std namespace).
 */
KOKKOS_INLINE_FUNCTION
void myswap(real_t& a, real_t& b)
{
    real_t c = a;
    a = b;
    b = c;
}

#endif // KOKKOS_SHARED_H_
