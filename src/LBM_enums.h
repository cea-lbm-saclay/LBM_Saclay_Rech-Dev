#ifndef LBM_ENUMS_H_
#define LBM_ENUMS_H_

//! dimension of the problem
enum DimensionType { TWO_D = 2, THREE_D = 3, DIM2 = 2, DIM3 = 3 };

//! enumerate all LBM lattice available
enum LBM_Lattice_t { D2Q5, D2Q9, D3Q7, D3Q15, D3Q19, D3Q27 };

//! enumerate number of population (direction) for a given lattice
enum Npop_t {
  NPOP_D2Q5 = 5,
  NPOP_D2Q9 = 9,
  NPOP_D3Q7 = 7,
  NPOP_D3Q15 = 15,
  NPOP_D3Q19 = 19,
  NPOP_D3Q27 = 27
};

//! enumerate LBM physics module
enum LBM_physics_t {
  LBM_FLUID,
  LBM_TRANSPORT,
  LBM_HEAT_TRANSFER,
  LBM_PHASE_FIELD,
  LBM_FLUID_PF_COUPLING
};

//! face index
enum FaceIdType {
  FACE_XMIN,
  FACE_XMAX,
  FACE_YMIN,
  FACE_YMAX,
  FACE_ZMIN,
  FACE_ZMAX
};

//! type of boundary condition (note that BC_COPY is only used in the MPI
//! version for inside boundary)
enum BoundaryConditionType {

  BC_COPY,      /*!< only used in MPI parallelized version, for communicating
                   internal borders / ghosts */
  BC_PERIODIC,  /*!< periodic border condition */
  BC_ZERO_FLUX, /*!< zero-flux at cell interface  */
  BC_ANTI_BOUNCE_BACK, /*!< anti bounce back for dirichlet cdt at cell interface
                        */
  BC_BOUNCEBACK,
  BC_POISEUILLE,
  BC_FREEFLOW,
  BC_UNDEFINED,

};

enum BoundaryFieldValues {
  BOUNDARY_VELOCITY_X,
  BOUNDARY_VELOCITY_Y,
  BOUNDARY_VELOCITY_Z,
  BOUNDARY_PRESSURE,
  BOUNDARY_PHASE_FIELD,
  BOUNDARY_CONCENTRATION,
  BOUNDARY_CONCENTRATIONB,
  BOUNDARY_VALUES_COUNT,
};
enum BoundaryField {
  BOUNDARY_EQUATION_1,
  BOUNDARY_EQUATION_2,
  BOUNDARY_EQUATION_3,
  BOUNDARY_EQUATION_4,
  BOUNDARY_EQUATION_5,
  BOUNDARY_COUNT
};
//! enum component index
enum ComponentIndex2D { IX = 0, IY = 1, IZ = 2 };

//! direction used in directional splitting scheme
enum Direction {
  XDIR = 0,
  YDIR = 1,
  ZDIR = 2,
  DIR_X = 0,
  DIR_Y = 1,
  DIR_Z = 2
};

//! location of the outside boundary
enum BoundaryLocation {
  XMIN = 0,
  XMAX = 1,
  YMIN = 2,
  YMAX = 3,
  ZMIN = 4,
  ZMAX = 5
};

//! problem type
enum ProblemType { PROBLEM_GAUSSIAN_HILL };

//! collision types
enum CollisionTypes { BGK, TRT, MRT };

//! collision types
enum TRT_tauS_method { FIXED_TAU, FIXED_LAMBDA, CONDITIONAL_TAU };

#endif // LBM_ENUMS_H_
