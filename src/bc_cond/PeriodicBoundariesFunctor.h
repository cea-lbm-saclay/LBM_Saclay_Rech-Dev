#ifndef BORDER_PERIODIC_FUNCTORS_H_
#define BORDER_PERIODIC_FUNCTORS_H_

#include <limits> // for std::numeric_limits
#ifdef __CUDA_ARCH__
#include <math_constants.h>
#endif // __CUDA_ARCH__

#include "LBMParams.h"
#include "LBM_Base_Functor.h"
#include "kokkos_shared.h"

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Apply periodic boundary conditions - 2D.
 *
 *
 * Face numbering:
 *
 *      4
 *    ____
 *   |    |
 * 1 |    | 3
 *   |____|
 *
 *     2
 *
 *
 * Speed numbering:
 *
 * D2Q5
 *       2
 *       |
 *   3  - - 1
 *       |
 *       4
 *
 * D2Q9
 *
 * 6    2    5
 *      |
 * 3   -0-   1
 *      |
 * 7    4    8
 *
 */
template <int npop,
    FaceIdType faceId>
class MakeBoundariesFunctor2d_Periodic {

public:
    using LBMArray = typename LBMBaseFunctor<2, npop>::LBMArray;
    using FArray = typename LBMBaseFunctor<2, npop>::FArray;

    MakeBoundariesFunctor2d_Periodic(LBMParams params,
        FArray f)
        : params(params)
        , f(f) {};

    // static method which does it all: create and execute functor
    static void apply(LBMParams params,
        FArray f)
    {
        int max_size = std::max(params.isize,
            params.jsize);
        int nbIter = params.ghostWidth * max_size;

        MakeBoundariesFunctor2d_Periodic<npop, faceId> functor(params, f);
        Kokkos::parallel_for(nbIter, functor);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& index) const
    {
        const int nx = params.nx;
        const int ny = params.ny;

        //const int isize = params.isize;
        //const int jsize = params.jsize;
        const int ghostWidth = params.ghostWidth;

        const int imin = params.imin;
        const int imax = params.imax;

        const int jmin = params.jmin;
        const int jmax = params.jmax;

        int i, j;

        if (faceId == FACE_XMIN) {

            j = index / ghostWidth;
            i = index - j * ghostWidth;
            //~ i = index / ghostWidth;
            //~ j = index - i*ghostWidth;

            if (j >= jmin and j <= jmax and i >= 0 and i < ghostWidth) {

                for (int ipop = 0; ipop < npop; ++ipop) {
                    f(i, j, ipop) = f(i + nx, j, ipop);
                }
            }
        } // end FACE_XMIN

        if (faceId == FACE_XMAX) {

            j = index / ghostWidth;
            i = index - j * ghostWidth;
            //~ i = index / ghostWidth;
            //~ j = index - i*ghostWidth;
            i += (nx + ghostWidth);

            if (j >= jmin and j <= jmax and i >= nx + ghostWidth and i <= nx + 2 * ghostWidth - 1) {

                for (int ipop = 0; ipop < npop; ++ipop) {
                    f(i, j, ipop) = f(i - nx, j, ipop);
                }
            }
        } // end FACE_XMAX

        if (faceId == FACE_YMIN) {

            i = index / ghostWidth;
            j = index - i * ghostWidth;

            if (i >= imin and i <= imax and j >= 0 and j < ghostWidth) {

                for (int ipop = 0; ipop < npop; ++ipop) {
                    f(i, j, ipop) = f(i, j + ny, ipop);
                }
            }
        } // end FACE_YMIN

        if (faceId == FACE_YMAX) {

            i = index / ghostWidth;
            j = index - i * ghostWidth;
            j += (ny + ghostWidth);

            if (i >= imin and i <= imax and j >= ny + ghostWidth and j <= ny + 2 * ghostWidth - 1) {

                for (int ipop = 0; ipop < npop; ++ipop) {
                    f(i, j, ipop) = f(i, j - ny, ipop);
                }
            }
        } // end FACE_YMAX

    } // end operator ()

    LBMParams params;
    FArray f;

}; // MakeBoundariesFunctor2d_Periodic

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Apply periodic boundary conditions - 3D.
 *
 * Face numbering:
 *
 * 1 : left
 * 2 : front
 * 3 : right
 * 4 : behind
 * 5 : below
 * 6 : top
 *
 *     top = 6
 *           |
 *         _____
 *        /     /|
 *       /_____/ |
 *      |      | |  <- 3
 * 1 -> |   2  | /
 *      |______|/
 *
 *          |
 *  below = 5
 *
 *
 */
template <int npop,
    FaceIdType faceId>
class MakeBoundariesFunctor3d_Periodic {

public:
    using LBMArray = typename LBMBaseFunctor<3, npop>::LBMArray;
    using FArray = typename LBMBaseFunctor<3, npop>::FArray;

    MakeBoundariesFunctor3d_Periodic(LBMParams params,
        FArray f)
        : params(params)
        , f(f) {};

    // static method which does it all: create and execute functor
    static void apply(LBMParams params,
        FArray f)
    {
        int max_size = std::max(std::max(params.isize,
                                    params.jsize),
            params.ksize);

        int nbIter = params.ghostWidth * max_size * max_size;

        MakeBoundariesFunctor3d_Periodic<npop, faceId> functor(params, f);
        Kokkos::parallel_for(nbIter, functor);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& index) const
    {
        const int nx = params.nx;
        const int ny = params.ny;
        const int nz = params.nz;

        const int isize = params.isize;
        const int jsize = params.jsize;
        const int ghostWidth = params.ghostWidth;

        const int imin = params.imin;
        const int imax = params.imax;

        const int jmin = params.jmin;
        const int jmax = params.jmax;

        const int kmin = params.kmin;
        const int kmax = params.kmax;

        int i, j, k;

        if (faceId == FACE_XMIN) {

            // boundary xmin (index = i + j * ghostWidth + k * ghostWidth*jsize)
            k = index / (ghostWidth * jsize);
            j = (index - k * ghostWidth * jsize) / ghostWidth;
            i = index - j * ghostWidth - k * ghostWidth * jsize;

            if (k >= kmin and k <= kmax and j >= jmin and j <= jmax and i >= 0 and i < ghostWidth) {

                for (int ipop = 0; ipop < npop; ++ipop) {
                    f(i, j, k, ipop) = f(i + nx, j, k, ipop);
                }
            }
        } // end faceId == FACE_XMIN

        if (faceId == FACE_XMAX) {

            // boundary xmax (index = i + j *ghostWidth + k * ghostWidth*jsize)
            // same i,j,k as xmin, except translation along x-axis
            k = index / (ghostWidth * jsize);
            j = (index - k * ghostWidth * jsize) / ghostWidth;
            i = index - j * ghostWidth - k * ghostWidth * jsize;

            i += (nx + ghostWidth);

            if (k >= kmin and k <= kmax and j >= jmin and j <= jmax and i >= nx + ghostWidth and i <= nx + 2 * ghostWidth - 1) {

                for (int ipop = 0; ipop < npop; ++ipop) {
                    f(i, j, k, ipop) = f(i - nx, j, k, ipop);
                }
            }
        } // end faceId == FACE_XMAX

        if (faceId == FACE_YMIN) {

            // boundary ymin (index = i + j*isize + k*isize*ghostWidth)
            k = index / (isize * ghostWidth);
            j = (index - k * isize * ghostWidth) / isize;
            i = index - j * isize - k * isize * ghostWidth;

            if (i >= imin and i <= imax and j >= 0 and j < ghostWidth and k >= kmin and k <= kmax) {

                for (int ipop = 0; ipop < npop; ++ipop) {
                    f(i, j, k, ipop) = f(i, j + ny, k, ipop);
                }
            }
        } // end faceId == FACE_YMIN

        if (faceId == FACE_YMAX) {

            // boundary ymax (index = i + j*isize + k*isize*ghostWidth)
            // same i,j,k as ymin, except translation along y-axis
            k = index / (isize * ghostWidth);
            j = (index - k * isize * ghostWidth) / isize;
            i = index - j * isize - k * isize * ghostWidth;

            j += (ny + ghostWidth);

            if (k >= kmin and k <= kmax and j >= ny + ghostWidth and j <= ny + 2 * ghostWidth - 1 and i >= imin and i <= imax) {

                for (int ipop = 0; ipop < npop; ++ipop) {
                    f(i, j, k, ipop) = f(i, j - ny, k, ipop);
                }
            }
        } // end faceId == FACE_YMAX

        if (faceId == FACE_ZMIN) {

            // boundary zmin (index = i + j*isize + k*isize*jsize)
            k = index / (isize * jsize);
            j = (index - k * isize * jsize) / isize;
            i = index - j * isize - k * isize * jsize;

            //boundary_type = params.boundary_type_zmin;

            if (k >= 0 and k < ghostWidth and j >= jmin and j <= jmax and i >= imin and i <= imax) {

                for (int ipop = 0; ipop < npop; ++ipop) {
                    f(i, j, k, ipop) = f(i, j, k + nz, ipop);
                }
            }

        } // end faceId == FACE_ZMIN

        if (faceId == FACE_ZMAX) {

            // boundary zmax (index = i + j*isize + k*isize*jsize)
            // same i,j,k as ymin, except translation along y-axis
            k = index / (isize * jsize);
            j = (index - k * isize * jsize) / isize;
            i = index - j * isize - k * isize * jsize;

            k += (nz + ghostWidth);

            //boundary_type = params.boundary_type_zmax;

            if (k >= nz + ghostWidth and k <= nz + 2 * ghostWidth - 1 and j >= jmin and j <= jmax and i >= imin and i <= imax) {

                for (int ipop = 0; ipop < npop; ++ipop) {
                    f(i, j, k, ipop) = f(i, j, k - nz, ipop);
                }
            }

        } // end faceId == FACE_ZMAX

    } // end operator ()

    LBMParams params;
    FArray f;

}; // MakeBoundariesFunctor3d_Periodic

#endif // BORDER_PERIODIC_FUNCTORS_H_
