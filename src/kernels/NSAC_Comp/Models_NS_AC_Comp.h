#ifndef MODELS_NS_AC_C_H_
#define MODELS_NS_AC_C_H_
#include "InitConditionsTypes.h"
#include "Index_NS_AC_Comp.h"
#include <real_type.h>
namespace PBM_NS_AC_Compo {
// ================================================
//
// Models for mixt formulation of grand potential problems
//
// ================================================

struct ModelParams {

  using LBMState = typename Kokkos::Array<real_t, COMPONENT_SIZE>;
  static constexpr real_t NORMAL_EPSILON = 1.0e-16;

  void showParams() {
    if (myRank == 0) {
      std::cout << "W :    " << W << std::endl;
      std::cout << "tauphi :    " << 0.5 + (e2 * Mphi * dt / SQR(dx))
                << std::endl;
      std::cout << "tauNS :    " << (0.5 + (e2 * nu0 * dt / SQR(dx)))
                << std::endl;
      std::cout << "cs :    " << ((dx / dt) / SQRT(e2)) << std::endl;
      std::cout << "cs2 :    " << SQR(dx / dt) / e2 << std::endl;
      std::cout << "counter_term :    " << counter_term << std::endl;
      std::cout << "init :" << initType << "." << std::endl;
      std::cout << "Reynolds = " << SQRT(L * (-gz)) * L / nu0 << std::endl;
      std::cout << "Peclet = " << SQRT(L * (-gz)) * L / Mphi << std::endl;
      std::cout << "Capillaire = " << SQRT(L * (-gz)) * rho1 * nu1 / sigma
                << std::endl;
    }
  };

  ModelParams(){};
  ModelParams(const ConfigMap &configMap, LBMParams params) {
    T = params.tEnd;
    time = 0.0;
    dt = params.dt;
    dtprev = dt;
    dx = params.dx;
    e2 = configMap.getFloat("lbm", "e2", 3.0);
    gammaTRT1 = configMap.getFloat("equation1", "lambdaTRT", 1 / 12);
    gammaTRT2 = configMap.getFloat("equation2", "lambdaTRT", 1 / 12);
    fMach = configMap.getFloat("run", "fMach", 0.04);

    L = configMap.getFloat("mesh", "xmax", 1.0) -
        configMap.getFloat("mesh", "xmin", 0.0);
	y_dom_min = configMap.getFloat("mesh", "ymin", 1.0) ;
	y_dom_max = configMap.getFloat("mesh", "ymax", 1.0) ;
	
	// Read parameters of phase-field equation
    W      = configMap.getFloat("params", "W", 0.005);
    gamma  = configMap.getFloat("params", "gamma", 1.0);
    Mphi   = configMap.getFloat("params", "Mphi", 1.0);
    lambda = configMap.getFloat("params", "lambda", 0.0);
    sigma  = configMap.getFloat("params", "sigma", 0.0);

    D0 = configMap.getFloat("params", "D0", 1.0);
    D1 = configMap.getFloat("params", "D1", 1.0);

	// Read parameters of Navier-Stokes equation
    rho0 = configMap.getFloat("params", "rho0", 1.0);
    rho1 = configMap.getFloat("params", "rho1", 1.0);
    rho0_ini = configMap.getFloat("params", "rho0_ini", 1.0);
    rho0_eq = configMap.getFloat("params", "rho0_eq", 1.0);
    rho1_ini = configMap.getFloat("params", "rho1_ini", 1.0);
    rho1_eq = configMap.getFloat("params", "rho1_eq", 1.0);

    nu0 = configMap.getFloat("params", "nu0", 1.0);
    nu1 = configMap.getFloat("params", "nu1", 1.0);

    fx = configMap.getFloat("params", "fx", 0.0);
    fy = configMap.getFloat("params", "fy", 0.0);
    fz = configMap.getFloat("params", "fz", 0.0);
    gx = configMap.getFloat("params", "gx", 0.0);
    gy = configMap.getFloat("params", "gy", 0.0);
    gz = configMap.getFloat("params", "gz", 0.0);
    

    // Read parameters of composition equation
    c0_co = configMap.getFloat("params_composition", "c0_co", 0.0);
    c1_co = configMap.getFloat("params_composition", "c1_co", 0.0);
    mu_eq = configMap.getFloat("params_composition", "mu_eq", 0.0);
    c0_inf = configMap.getFloat("params_composition", "c0_inf", 0.0);
    c1_inf = configMap.getFloat("params_composition", "c1_inf", 0.0);

    mu0_inf = c0_inf - c0_co + mu_eq;
    mu1_inf = c1_inf - c1_co + mu_eq;

    //! Parameters 0 or 1
    PF_advec      = configMap.getInteger("params", "PF_advec"     , 0  );
    counter_term  = configMap.getFloat  ("params", "counter_term" , 1.0);
    cahn_hilliard = configMap.getInteger("params", "cahn_hilliard", 0  );
    
    // Initialization of Navier-Stokes equations
    Vx_init     = configMap.getFloat("init", "Vx_init_phi1" , 0.0);
    Vy_init     = configMap.getFloat("init", "Vy_init_phi1" , 0.0);
    Vz_init     = configMap.getFloat("init", "Vz_init_phi1" , 0.0);
    p_init_phi  = configMap.getFloat("init", "pressure_phi1", 0.0);
    p_init_phi2 = configMap.getFloat("init", "pressure_phi2", 0.0);

    // Initialization of phase-field: geometry of phi
    x0       = configMap.getFloat  ("init", "x0"         , 0.0);
    y0       = configMap.getFloat  ("init", "y0"         , 0.0);
    z0       = configMap.getFloat  ("init", "z0"         , 0.0);
    r0       = configMap.getFloat  ("init", "r0"         , 0.2);
    r1       = configMap.getFloat  ("init", "r1"         , 0.2);
	U0       = configMap.getFloat  ("init", "U0"         , 1.0);
    sign     = configMap.getFloat  ("init", "sign"       , 1  );
    // Initialization for Rayleigh-Taylor
    // commune
    ampl     = configMap.getFloat("init", "amplitude", 0.0);
    height_R = configMap.getFloat("init", "hauteur_R", 0.0);
    width_R  = configMap.getFloat("init", "largeur_R", 0.0);
    // sinusoïdale
    n_wave = configMap.getFloat("init", "nombre_onde", 1.0);
    wave_length = configMap.getFloat("init", "longueur_onde", 1.0);
    // exponentielle
    sigma0 = configMap.getFloat("init", "sigma0", 1.0);

    // Géométrie Two-Plane
    height = configMap.getFloat("init", "hauteur", 0.0);
    initVX_upper = configMap.getFloat("init", "initVX_upper", 0.0);
    initVX_lower = configMap.getFloat("init", "initVX_lower", 0.0);

    // Géométrie Bulle
    R  = configMap.getFloat("init", "rayon"  , 10.0);
    xC = configMap.getFloat("init", "Xcentre",  0.0);
    yC = configMap.getFloat("init", "Ycentre",  0.0);
    zC = configMap.getFloat("init", "Zcentre",  0.0);
    
    // Injection of fluid phase
    injector = configMap.getInteger("injector","injector", 0  );
	t_inject = configMap.getFloat  ("injector","t_inject", 0.0);
	q_inject = configMap.getFloat  ("injector","q_inject", 0.0);
	hx       = configMap.getFloat  ("injector","hx"      , 0.0);
    hy       = configMap.getFloat  ("injector","hy"      , 0.0);
    hz       = configMap.getFloat  ("injector","hz"      , 0.0);
    
    // Read parameters for solid phase psi
    solid_phase         = configMap.getInteger("init_solid","solid_phase"        , 0  );
    diffuse_bounce_back = configMap.getInteger("init_solid","diffuse_bounce_back", 0  );
    W_sol               = configMap.getFloat  ("init_solid","W_sol"              , 1.0);
    Mphi_sol            = configMap.getFloat  ("init_solid","Mphi_sol"           , 1.0);
	rho_sol             = configMap.getFloat  ("init_solid","rho_sol"            , 1.0);
	nu_sol              = configMap.getFloat  ("init_solid","nu_sol"             , 0.0);
	sign_sol            = configMap.getFloat  ("init_solid","sign_sol"   , 1.0);
	if (solid_phase == 0  ) Mphi_sol = Mphi ;
	if (nu_sol      == 0.0) nu_sol = nu1 ;
	// Paramètres pour l'angle de contact
	contact_angle     = configMap.getInteger("contact_angle","contact_angle", 0);
	Sigma12           = sigma ;
	Sigma13           = configMap.getFloat  ("contact_angle","sigma_phi1_solid", 0.0);
	Sigma23           = configMap.getFloat  ("contact_angle","sigma_phi2_solid", 0.0);
	// Solid geometry: sphere
	x0_solid_sphere   = configMap.getFloat("init_solid","x0_solid_circ" , 0.0);
	y0_solid_sphere   = configMap.getFloat("init_solid","y0_solid_circ" , 0.0);
	z0_solid_sphere   = configMap.getFloat("init_solid","z0_solid_circ" , 0.0);
	r0_solid_sphere   = configMap.getFloat("init_solid","r0_solid_circ" , 0.0);
	
	// Solid geometry: cylinder
	x0_solid_cylinder = configMap.getFloat("init_solid","x0_solid_cyl"  , 0.0);
	y0_solid_cylinder = configMap.getFloat("init_solid","y0_solid_cyl"  , 0.0);
	z0_solid_cylinder = configMap.getFloat("init_solid","z0_solid_cyl"  , 0.0);
	h0_solid_cylinder = configMap.getFloat("init_solid","h0_solid_cyl"  , 0.0);
	// cylinder of radius r0
	r0_solid_cylinder = configMap.getFloat("init_solid","r0_solid_cyl"  , 0.0);
	// with hole of radius r1
	r1_solid_cylinder = configMap.getFloat("init_solid","r1_solid_cyl"  , 0.0);
	// Second cylinder (above for injector or below for container) of radius r2
	z2_solid_cylinder = configMap.getFloat("init_solid","z2_solid_cyl"  , 0.0);
	r2_solid_cylinder = configMap.getFloat("init_solid","r2_solid_cyl"  , 0.0);
	h2_solid_cylinder = configMap.getFloat("init_solid","h2_solid_cyl"  , 0.0);
	
	// Solid geometry: rectangle
	x0_solid_rect     = configMap.getFloat("init_solid","x0_solid_rect"   , 0.0);
	y0_solid_rect     = configMap.getFloat("init_solid","y0_solid_rect"   , 0.0);
	z0_solid_rect     = configMap.getFloat("init_solid","z0_solid_rect"   , 0.0);
	Larg_solid_rect   = configMap.getFloat("init_solid","Larg_solid_rect" , 0.0);
	long_solid_rect   = configMap.getFloat("init_solid","long_solid_rect" , 0.0);
	haut_solid_rect   = configMap.getFloat("init_solid","haut_solid_rect" , 0.0);
	theta_solid_rect  = configMap.getFloat("init_solid","theta_solid_rect", 0.0);
	// Géométrie conteneur en U
	x0_gl             = configMap.getFloat("init_solid", "x0_gl"         , 0.0);
	y0_gl             = configMap.getFloat("init_solid", "y0_gl"         , 0.0);
	height_gl         = configMap.getFloat("init_solid", "height_gl"     , 0.0);
	width_gl          = configMap.getFloat("init_solid", "width_gl"      , 0.0);
	thickness_gl      = configMap.getFloat("init_solid", "thickness_gl"  , 0.0);
	h_glass           = configMap.getFloat("init_solid", "h_glass"       , 0.0);
	length_hole_gl    = configMap.getFloat("init_solid", "length_hole_gl", 0.0);
	// PARAMETRES TUBE TORRICELLI
	y0_tu        = configMap.getFloat("init_solid", "y0_tu"       , 0.0);
    height_tu    = configMap.getFloat("init_solid", "height_tu"   , 0.0);
    width_tu     = configMap.getFloat("init_solid", "width_tu"    , 0.0);
    thickness_tu = configMap.getFloat("init_solid", "thickness_tu", 0.0);
    
    // Paramètres pour un solide qui bouge
    move_solid               = configMap.getInteger("move_solid","move_solid"              , 0  );
	coeff_solid_penalization = configMap.getFloat  ("move_solid","coeff_solid_penalization", 0.0);
	oscil_solid_ampl         = configMap.getFloat  ("move_solid","oscil_solid_ampl"        , 0.0);
	oscil_solid_freq         = configMap.getFloat  ("move_solid","oscil_solid_freq"        , 0.0);
	x0_solid_rotate          = configMap.getFloat  ("move_solid","x0_solid_rotate"         , 0.0);
	y0_solid_rotate          = configMap.getFloat  ("move_solid","y0_solid_rotate"         , 0.0);
	rotation_freq            = configMap.getFloat  ("move_solid","rotation_freq"           , 0.0);
	if (solid_phase == 0) {
		lagrange_multiplier = 0 ;
	}
	else {
		lagrange_multiplier = 1 ;
	}
    //! Temps de relaxation MRT
    for (int i = 0; i < NPOP_D3Q27; i++) {
      std::string taui = "tau" + std::to_string(i);
      tauMatrixAC[i] = configMap.getFloat("MRT_AC", taui, -1.0);
    }

    for (int i = 0; i < NPOP_D3Q27; i++) {
      std::string taui = "tau" + std::to_string(i);
      tauMatrixNS[i] = configMap.getFloat("MRT_NS", taui, -1.0);
    }

    for (int i = 0; i < NPOP_D3Q27; i++) {
      std::string taui = "tau" + std::to_string(i);
      tauMatrixComp[i] = configMap.getFloat("MRT_COMP", taui, -1.0);
    }

    // Initialisation type de la condition initiale
    initType = PHASE_FIELD_INIT_UNDEFINED;
    std::string initTypeStr =
        std::string(configMap.getString("init", "init_type", "unknown"));

    if (initTypeStr == "vertical")
      initType = PHASE_FIELD_INIT_VERTICAL;
    else if (initTypeStr == "sphere")
      initType = PHASE_FIELD_INIT_SPHERE;
    else if (initTypeStr == "cylinder")
      initType = PHASE_FIELD_INIT_CYLINDER;
    else if (initTypeStr == "square")
      initType = PHASE_FIELD_INIT_SQUARE;
    else if (initTypeStr == "data")
      initType = PHASE_FIELD_INIT_DATA;
    else if (initTypeStr == "2sphere")
      initType = PHASE_FIELD_INIT_2SPHERE;
    else if (initTypeStr == "rayleigh_taylor")
      initType = PHASE_FIELD_INIT_TAYLOR;
    else if (initTypeStr == "two_plane")
      initType = TWO_PLANE;
    else if (initTypeStr == "bubble_taylor")
      initType = BUBBLE_TAYLOR;
    else if (initTypeStr == "serpentine")
      initType = TWO_PHASE_SERPENTINE;
    else if (initTypeStr == "zalesak")
      initType = TWO_PHASE_ZALESAK;
    else if (initTypeStr == "rising_bubble")
      initType = RISING_BUBBLE;
    else if (initTypeStr == "splash")
      initType = TWO_PHASE_SPLASH;
    else if (initTypeStr == "breaking_wave")
      initType = TWO_PHASE_BREAKING_WAVE;
    else if (initTypeStr == "spinodal_decomposition")
	  initType = TWO_PHASE_SPINODAL_DECOMP;
    else if (initTypeStr == "dam_break")
      initType = DAM_BREAK;
    else if (initTypeStr == "taylor_expo")
      initType = PHASE_FIELD_INIT_TAYLOR_PERTURB_EXPO;
    else if (initTypeStr == "container")
      initType = PHASE_FIELD_GLASS;
    else if (initTypeStr == "torricelli")
      initType = PHASE_FIELD_TORRICELLI ;
      
    initTypeSol = SOLID_INIT_NO_SOLID; 
    std::string initTypeSolStr =
        std::string(configMap.getString("init_solid", "init_solid_type", "no_solid"));
    
    if (initTypeSolStr == "sphere")
	  initTypeSol = SOLID_INIT_SPHERE;
	else if (initTypeSolStr == "cylinder")
      initTypeSol = SOLID_INIT_CYLINDER;
    else if (initTypeSolStr == "cylinder_tank")
      initTypeSol = SOLID_INIT_CYLINDER_TANK;
    else if (initTypeSolStr == "rectangle")
      initTypeSol = SOLID_INIT_RECTANGLE;
    else if (initTypeSolStr == "container")
      initTypeSol = SOLID_INIT_GLASS;
    else if (initTypeSolStr == "container_hole")
      initTypeSol = SOLID_INIT_GLASS_HOLE;
    else if (initTypeSolStr == "torricelli")
      initTypeSol = SOLID_INIT_TORRICELLI ;
    
    initMoveSolid = SOLID_NO_MOVE ;
    std::string initMoveSolidStr = 
        std::string(configMap.getString("move_solid", "move_solid_type", "no_move"));
    if (initMoveSolidStr == "oscillation")
	  initMoveSolid = SOLID_MOVE_OSCILL;
	else if (initMoveSolidStr == "rotation")
	  initMoveSolid = SOLID_MOVE_ROTATE;
    
    myRank = 0;
#ifdef USE_MPI
    myRank = params.communicator->getRank();
#endif
    showParams();
  }

  //! model params
  //! =====================================    Main parameters  =========================================
  int PF_advec, cahn_hilliard ;
  int initType;
  real_t dx, dt, dtprev, e2, cs2, fMach, L;
  real_t time, T, period;
  real_t gammaTRT1, gammaTRT2; // trt params
  //! Input parameters for phase-field equation
  real_t W, Mphi, lambda, gamma, counter_term ;
  //! Input parameters for Navier-Stokes equations
  real_t U0, nu1, rho1, nu0, rho0, sigma;
  real_t rho0_ini, rho0_eq, rho1_ini, rho1_eq;
  real_t fx, fy, fz, gx, gy, gz, hx, hy, hz;
  //! Input parameters for composition equation
  real_t c0_co, c1_co, c0_inf, c1_inf, mu_eq, mu0_inf, mu1_inf, D0, D1;
  //! =====================================    initialization   =========================================
  real_t x0, y0, z0, r0, r1, sign, y_dom_min, y_dom_max ;
  real_t ampl, height_R, width_R, width, height, n_wave, wave_length, sigma0;
  real_t Vx_init, Vy_init, Vz_init, p_init_phi, p_init_phi2, t_inject, q_inject ;
  real_t initVX_upper, initVX_lower;
  real_t xC, yC, zC, R;
  //! ===================================== init solid phase =========================================
  int solid_phase, move_solid, initTypeSol, initMoveSolid;
  int diffuse_bounce_back, lagrange_multiplier, contact_angle, injector;
  real_t Mphi_sol, W_sol, rho_sol, nu_sol, sign_sol, coeff_solid_penalization;
  real_t Sigma12, Sigma13, Sigma23;
  real_t r0_solid_sphere, x0_solid_sphere, y0_solid_sphere, z0_solid_sphere ;
  real_t x0_solid_cylinder, y0_solid_cylinder, z0_solid_cylinder, r0_solid_cylinder, h0_solid_cylinder, r1_solid_cylinder ;
  real_t z2_solid_cylinder, r2_solid_cylinder,  h2_solid_cylinder;
  real_t x0_solid_rect, y0_solid_rect, z0_solid_rect, Larg_solid_rect, long_solid_rect, haut_solid_rect, theta_solid_rect, rotation_freq ;
  real_t x0_gl, y0_gl, height_gl, width_gl, thickness_gl, h_glass, length_hole_gl;
  real_t y0_tu, height_tu, width_tu, thickness_tu;
  real_t oscil_solid_ampl, oscil_solid_freq, x0_solid_rotate, y0_solid_rotate ;
  //! MRT params
  real_t tauMatrixAC[27], tauMatrixNS[27], tauMatrixComp[27];

  
  
  
  int myRank; // used during mpi simulations to know whether to output params

  // equation of the interface
  KOKKOS_INLINE_FUNCTION real_t phi0(real_t x) const {
    return 0.5 * (1 + tanh(sign * 2.0 * x / W));
  }
  KOKKOS_INLINE_FUNCTION real_t psi0(real_t x) const {
    return 0.5 * (1 + tanh(sign_sol * 2.0 * x / W_sol));
  }
  KOKKOS_INLINE_FUNCTION real_t c0(real_t x) const {
    return 0.5 *
           (c1_inf + c0_inf + (c1_inf - c0_inf) * tanh(sign * 2.0 * x / W));
  }
  // interpolation of rho
  KOKKOS_INLINE_FUNCTION real_t hrho(real_t phi) const {
    return SQR(phi) * (3.0 - 2.0 * phi);
  }
  KOKKOS_INLINE_FUNCTION real_t hrho_prime(real_t phi) const {
    return 6.0 * phi * (1.0 - phi);
  }
  // KOKKOS_INLINE_FUNCTION
  KOKKOS_INLINE_FUNCTION real_t rho0c(real_t c) const {
    return (rho0_eq - rho0_ini) / (c0_co - c0_inf) * (c - c0_inf) + rho0_ini;
  }
  KOKKOS_INLINE_FUNCTION real_t rho1c(real_t c) const {
    return (rho1_eq - rho1_ini) / (c1_co - c1_inf) * (c - c1_inf) + rho1_ini;
  }
  KOKKOS_INLINE_FUNCTION real_t rho0c_prime(real_t c) const {
    return (rho0_eq - rho0_ini) / (c0_co - c0_inf);
  }
  KOKKOS_INLINE_FUNCTION real_t rho1c_prime(real_t c) const {
    return (rho1_eq - rho1_ini) / (c1_co - c1_inf);
  }
  KOKKOS_INLINE_FUNCTION real_t interp_rho(real_t phi, real_t c) const {
    return (1.0 - hrho(phi)) * rho0c(c) + hrho(phi) * rho1c(c);
  }
  
  KOKKOS_INLINE_FUNCTION real_t interp_rho_with_phi_psi (real_t phi, real_t psi, real_t phi2) const {
	   real_t density = psi*rho_sol + phi*rho1 + phi2*rho0;
    return density ;
  }
  // interpolation functions
  KOKKOS_INLINE_FUNCTION real_t p(real_t phi) const {
    return SQR(phi) * (3.0 - 2.0 * phi);
  }
  KOKKOS_INLINE_FUNCTION real_t p_prime(real_t phi) const {
    return 6.0 * phi * (1.0 - phi);
  }
  KOKKOS_INLINE_FUNCTION real_t g_prime(real_t phi) const {
    return 16.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi);
  }
  KOKKOS_INLINE_FUNCTION real_t g2_prime(real_t phi) const {
    return phi * (1.0 - phi) * (1.0 - 2.0 * phi);
  }
  // interpolation function for c_co
  KOKKOS_INLINE_FUNCTION real_t h(real_t phi) const { return phi; }
  KOKKOS_INLINE_FUNCTION real_t h_prime(real_t phi) const { return 1.0; }
  KOKKOS_INLINE_FUNCTION real_t c_co(real_t phi) const {
    return c1_co * h(phi) + c0_co * (1 - h(phi));
  }
  // ingerpolation function for D0 and D1
  KOKKOS_INLINE_FUNCTION real_t q(real_t phi) const { return phi; }
  KOKKOS_INLINE_FUNCTION real_t q_prime(real_t phi) const { return 1.0; }
  KOKKOS_INLINE_FUNCTION real_t interpol_D(real_t phi) const {
    return D1 * q(phi) + D0 * (1 - q(phi));
  }

  // ================================================
  //
  // functions for Fakhari et. al.'s model
  //
  // ================================================

  struct TagCompo {};
  struct TagSansGamma {};

  // ==============================================================================================================
  // 					                        Models for phase field
  // ==============================================================================================================

  // =======================================================
  // relaxation coef for LBM scheme of phase field equation
  KOKKOS_INLINE_FUNCTION
  real_t tau_PHI(EquationTag1 tag, const LBMState &lbmState) const {
    real_t tau = 0.5 + (e2 * Mphi * dt / SQR(dx));
    return (tau);
  }

  KOKKOS_INLINE_FUNCTION
  real_t tau_PHI_sol (EquationTag1 tag, const LBMState &lbmState) const {
    real_t Mobility = (Mphi_sol*lbmState[IPSI] + Mphi*(1-lbmState[IPSI])) ;
    real_t tau      = 0.5 + (e2 * Mobility * dt / SQR(dx));
    return (tau);
  }
  // =======================================================
  // zero order moment for phase field
  KOKKOS_INLINE_FUNCTION
  real_t M0_PHI(EquationTag1 tag, const LBMState &lbmState) const {
    return lbmState[IPHI];
  }

  // =======================================================
  // first order moment for phase field
  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> M1_PHI(EquationTag1 tag,
                                           const LBMState &lbmState) const {
    RVect<dim> term;
    term[IX] = lbmState[IPHI] * lbmState[IU];
    term[IY] = lbmState[IPHI] * lbmState[IV];
    if (dim == 3) {
      term[IZ] = lbmState[IPHI] * lbmState[IW];
    }
    return term;
  }

  // =======================================================
  // second order moment for phase field
  KOKKOS_INLINE_FUNCTION
  real_t M2_PHI(EquationTag1 tag, const LBMState &lbmState) const {
    return lbmState[IPHI];
  }
  
  // second order moment for phase field
  KOKKOS_INLINE_FUNCTION
  real_t M2_MU_PHI(EquationTag1 tag, const LBMState &lbmState) const {
	const real_t mu_phi = sigma * 1.5 / W *
        (g_prime(lbmState[IPHI]) - SQR(W) * lbmState[ILAPLAPHI]);
    return mu_phi;
  }

  // =======================================================
  // Sources term for phase field
  //========================================================
  // Derivative of double-well
  KOKKOS_INLINE_FUNCTION
  real_t S_dw(EquationTag1 tag, const LBMState &lbmState) const {
    real_t val_noct = -(1.0 - counter_term) * Mphi / SQR(W) * 16.0 *
                         lbmState[IPHI] * (1.0 - lbmState[IPHI]) *
                         (1.0 - 2.0 * lbmState[IPHI]);
    real_t source_noct = (1-cahn_hilliard)*val_noct ;
    return source_noct;
  }
  // =======================================================
  // Counter term for Conservative Allen-Cahn
  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> S_ct(const LBMState &lbmState) const {
    const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY]) +
                             SQR(lbmState[IDPHIDZ])) +
                        NORMAL_EPSILON;
    real_t val_ct = Mphi * counter_term * 4.0 * lbmState[IPHI] *
                      (1.0 - lbmState[IPHI]) / (W * norm);

	real_t force_ct = (1-cahn_hilliard)*val_ct ;
    RVect<dim> term;
    term[IX] = force_ct * lbmState[IDPHIDX];
    term[IY] = force_ct * lbmState[IDPHIDY];
    if (dim == 3) {
      term[IZ] = force_ct * lbmState[IDPHIDZ];
    }
    return term;
  }
  
  // =======================================================
  // Counter term for Conservative Allen-Cahn
  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> S_ct_psi(const LBMState &lbmState) const {
    const real_t norm = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY]) +
                             SQR(lbmState[IDPHIDZ])) +
                        NORMAL_EPSILON;
    real_t factr  = Mphi_sol*lbmState[IPSI] + Mphi*(1.-lbmState[IPSI]) ;
    real_t val_ct = factr * counter_term * 4.0 * lbmState[IPHI] *
                      (1.0 - lbmState[IPHI]) / (W * norm);

	real_t force_ct = (1-cahn_hilliard)*val_ct ;
    RVect<dim> term;
    term[IX] = force_ct * lbmState[IDPHIDX];
    term[IY] = force_ct * lbmState[IDPHIDY];
    if (dim == 3) {
      term[IZ] = force_ct * lbmState[IDPHIDZ];
    }
    return term;
  }
  // =======================================================
  // Source term for Allen-Cahn coupled with diffusion
  KOKKOS_INLINE_FUNCTION
  real_t S_st(EquationTag1 tag, const LBMState &lbmState) const {
    real_t S = p_prime(lbmState[IPHI]) * (c0_co - c1_co) *
               (mu(lbmState[IC], lbmState[IPHI]) - mu_eq);
    return -lambda * Mphi / SQR(W) * S;
  }
  // =======================================================
  // Source term for injection in conservative Allen-Cahn
  KOKKOS_INLINE_FUNCTION
  real_t Update_S_injection (const LBMState &lbmState) const {
	real_t coeff = injector * q_inject * 0.5*(1.0+tanh(2.0*(t_inject - time))) ;
    real_t Sinject = coeff * lbmState[ISPHI] ;
    return Sinject;
  }
  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> Vect_inject(const LBMState &lbmState) const {
    real_t coeff   = injector * q_inject * 0.5*(1.0+tanh(2.0*(t_inject - time))) ;
    real_t Sinject = coeff * lbmState[ISPHI] ;
    RVect<dim> S_imp ;
    S_imp[IX] = Sinject * hx ;
    S_imp[IY] = Sinject * hy ;
    S_imp[IZ] = Sinject * hz ;
    return S_imp;
  }
  // =======================================================
  // Lagrange multiplier for solid interaction in conserv Allen-Cahn Equation
  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> Lagrange_mult (const LBMState &lbmState) const {
    const real_t norm_phi1 = sqrt(SQR(lbmState[IDPHIDX]) + SQR(lbmState[IDPHIDY]) +
                             SQR(lbmState[IDPHIDZ])) +   NORMAL_EPSILON;
     
    const real_t norm_phi2 = sqrt(SQR(lbmState[IDPHI2DX]) + SQR(lbmState[IDPHI2DY]) +
                             SQR(lbmState[IDPHI2DZ])) +   NORMAL_EPSILON;
    
    const real_t norm_psi = sqrt(SQR(lbmState[IDPSIDX]) + SQR(lbmState[IDPSIDY]) +
                             SQR(lbmState[IDPSIDZ])) +   NORMAL_EPSILON;                        
                             
    real_t CT_phi1 = counter_term * 4.0 * lbmState[IPHI] *
                      (1.0 - lbmState[IPHI]) / (W * norm_phi1);
                      
    real_t CT_phi2 = counter_term * 4.0 * lbmState[IPHI2] *
                      (1.0 - lbmState[IPHI2]) / (W * norm_phi2);
                      
    real_t CT_psi  = counter_term * 4.0 * lbmState[IPSI] *
                      (1.0 - lbmState[IPSI]) / (W_sol * norm_psi);
                      
    real_t factr   = solid_phase * 0.5 * (Mphi_sol*lbmState[IPSI] + Mphi*(1.-lbmState[IPSI])) ;

    RVect<dim> lagr_mult ;
    lagr_mult[IX] = factr * (CT_phi1*lbmState[IDPHIDX] + CT_phi2*lbmState[IDPHI2DX] + CT_psi*lbmState[IDPSIDX]);
	lagr_mult[IY] = factr * (CT_phi1*lbmState[IDPHIDY] + CT_phi2*lbmState[IDPHI2DY] + CT_psi*lbmState[IDPSIDY]);
    if (dim == 3) {
      lagr_mult[IZ] = factr * (CT_phi1 * lbmState[IDPHIDZ] + CT_phi2 * lbmState[IDPHI2DZ] + CT_psi*lbmState[IDPSIDZ]);
    }
    return lagr_mult;
  }
  // ==============================================================================================================
  // 					                        Models for solid phase psi
  // ==============================================================================================================
  //
  // =======================================================
  // Definition of oscillating solid velocity
  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> Oscill_solid_velocity(const LBMState &lbmState) const {
	real_t ampl_sol = oscil_solid_ampl ;
	real_t freq_sol = oscil_solid_freq ;
	real_t UB_X     = ampl_sol * freq_sol * cos((time+dt)*freq_sol);
	real_t UB_Y     = 0.0 ;
	real_t UB_Z     = 0.0 ;
	RVect<dim> Solid_Velocity ;
	Solid_Velocity[IX] = UB_X ;
	Solid_Velocity[IY] = UB_Y ;
	if (dim == 3) {
		Solid_Velocity[IZ] = UB_Z ;
	}
    return Solid_Velocity ;
  }
  // =======================================================
  // Definition of rotating solid velocity
  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> Rotate_solid_velocity(real_t x, real_t y, const LBMState &lbmState) const {
	//real_t ampl_sol = oscil_solid_ampl ;
	//real_t freq_sol = oscil_solid_freq ;
	real_t UB_X     = rotation_freq * (y - y0_solid_rotate);
	real_t UB_Y     = rotation_freq * (x0_solid_rotate - x);
	real_t UB_Z     = 0.0 ;
	RVect<dim> Solid_Velocity ;
	Solid_Velocity[IX] = UB_X ;
	Solid_Velocity[IY] = UB_Y ;
	if (dim == 3) {
		Solid_Velocity[IZ] = UB_Z ;
	}
    return Solid_Velocity ;
  }
  //
  // ===========================================================================================
  // Initialize psi as a sphere
  KOKKOS_INLINE_FUNCTION
  real_t psi_init_sphere(real_t x, real_t y, real_t z) const {
	  real_t xpsi = (r0_solid_sphere - sqrt(SQR(x - x0_solid_sphere) +
								   SQR(y - y0_solid_sphere) +
								   SQR(y - z0_solid_sphere)  ));
	  return xpsi;
  }
  // Initialize psi as a cylinder
  KOKKOS_INLINE_FUNCTION
  real_t psi_init_cylinder3D(real_t x, real_t y, real_t z) const {
	real_t xpsi = 0.0 ;
	real_t d1   = r0_solid_cylinder - sqrt(SQR(x - x0_solid_cylinder) + SQR(y - y0_solid_cylinder)) ;
	real_t d2   = r1_solid_cylinder - sqrt(SQR(z - z0_solid_cylinder)) ;
	xpsi = FMIN(d1,d2);
	return xpsi;
  }
  // Initialize psi as a cylinder tank
  KOKKOS_INLINE_FUNCTION
  real_t psi_init_cylinder_tank3D(real_t x, real_t y, real_t z) const {
	real_t xpsi = 0.0 ;
	real_t z0 = z0_solid_cylinder ;
	real_t z2 = z2_solid_cylinder ;
	real_t h0 = h0_solid_cylinder ;
	real_t h2 = h2_solid_cylinder ;
	real_t d1   = sqrt(SQR(x - x0_solid_cylinder) + SQR(y - y0_solid_cylinder)) ;
	real_t d2   = h0 - sqrt(SQR(z - z0)) ;
	real_t d3   = h2 - sqrt(SQR(z - z2)) ;
	if (z >= (z0-h0)) {
		xpsi = FMIN(FMIN(d1 - r1_solid_cylinder, -d1 + r0_solid_cylinder),d2) ;
	}
	else if (z < (z0-h0)) {
		xpsi = FMIN(FMIN(d1 - r2_solid_cylinder, -d1 + r0_solid_cylinder),d3) ;
	}
	return xpsi;
  }
  // Initialize psi as a rectangle 2D
  KOKKOS_INLINE_FUNCTION
  real_t psi_init_Rectangle2D(real_t x, real_t y) const {
	real_t xpsi = 0.0 ;
	real_t x0    = x0_solid_rect;
	real_t y0    = y0_solid_rect;
	real_t L     = Larg_solid_rect;
	real_t l     = long_solid_rect;
	real_t theta = theta_solid_rect ;
    real_t xr    = cos( theta)*(x-x0) - sin(-theta)*(y-y0) + x0 ;
    real_t yr    = sin(-theta)*(x-x0) + cos( theta)*(y-y0) + y0 ;
    if ( sqrt(SQR(xr-x0)) <= l/2 or sqrt(SQR(yr-y0)) <= L/2 ){
         xpsi = FMIN(l/2 - sqrt(SQR(x0-xr)),L/2 - sqrt(SQR(y0-yr)));
    } else{
		if (yr > y0 + L/2){
			if(xr < x0 -l/2){
				xpsi = -sqrt(SQR(xr - (x0-l/2)) + SQR(yr - (y0+L/2)));
			}
            if(xr > x0 + l/2){
                xpsi = -sqrt(SQR(xr - (x0+l/2)) + SQR(yr - (y0+L/2)));
            }
		}
        if (yr < y0 - L/2){
            if(xr < x0 -l/2){
				xpsi = -sqrt(SQR(xr - (x0-l/2)) + SQR(yr - (y0-L/2)));
			}
			if(xr > x0 + l/2){
				xpsi = -sqrt(SQR(xr - (x0+l/2)) + SQR(yr - (y0-L/2)));
            }
		}
	}
    return xpsi;
  }
  
  // Initialize psi as a rectangle 3D
  KOKKOS_INLINE_FUNCTION
  real_t psi_init_Rectangle3D(real_t x, real_t y, real_t z) const {
	real_t xpsi = 0.0 ;
	real_t x0    = x0_solid_rect;
    real_t y0    = y0_solid_rect;
    real_t z0    = z0_solid_rect;
    real_t L     = Larg_solid_rect;
    real_t l     = long_solid_rect;
    real_t h     = haut_solid_rect;
    real_t theta = theta_solid_rect ;
    real_t xr    = cos( theta)*(x-x0) - sin(-theta)*(y-y0) + x0 ;
    real_t yr    = y ;
    real_t zr    = sin(-theta)*(x-x0) + cos( theta)*(z-z0) + z0 ;
    
    if ( sqrt(SQR(xr-x0)) <= l/2 or sqrt(SQR(yr-y0)) <= L/2 or sqrt(SQR(zr-z0))){
         xpsi = FMIN(FMIN(l/2 - sqrt(SQR(x0-xr)),L/2 - sqrt(SQR(y0-yr))),h/2 - sqrt(SQR(z0-zr)));
    } else{
		if (zr > z+h/2) {
			if (yr > y0 + L/2){
				if(xr < x0 -l/2){
					xpsi = -sqrt(SQR(xr - (x0-l/2)) + SQR(yr - (y0+L/2)) + SQR(zr -(z0+h/2)));
				}
				if(xr > x0 + l/2){
					xpsi = -sqrt(SQR(xr - (x0+l/2)) + SQR(yr - (y0+L/2)) + SQR(zr -(z0+h/2)));
				}
			}
			if (yr < y0 - L/2){
				if(xr < x0 -l/2){
					xpsi = -sqrt(SQR(xr - (x0-l/2)) + SQR(yr - (y0-L/2)) + SQR(zr -(z0+h/2)));
				}
				if(xr > x0 + l/2){
					xpsi = -sqrt(SQR(xr - (x0+l/2)) + SQR(yr - (y0-L/2)) + SQR(zr -(z0+h/2)));
				}
			}
		}
		if (zr < z - h/2) {
			if (yr > y0 + L/2){
				if(xr < x0 -l/2){
					xpsi = -sqrt(SQR(xr - (x0-l/2)) + SQR(yr - (y0+L/2)) + SQR(zr -(z0-h/2)));
				}
				if(xr > x0 + l/2){
					xpsi = -sqrt(SQR(xr - (x0+l/2)) + SQR(yr - (y0+L/2)) + SQR(zr -(z0-h/2)));
				}
			}
			if (yr < y0 - L/2){
				if(xr < x0 -l/2){
					xpsi = -sqrt(SQR(xr - (x0-l/2)) + SQR(yr - (y0-L/2)) + SQR(zr -(z0-h/2)));
				}
				if(xr > x0 + l/2){
					xpsi = -sqrt(SQR(xr - (x0+l/2)) + SQR(yr - (y0-L/2)) + SQR(zr -(z0-h/2)));
				}
			}
		}
	}
    return xpsi;
  }
  // Update psi_new with psi_old - us.grad(psi)*dt + (dt2/2)us2laplacien(psi)
  KOKKOS_INLINE_FUNCTION
  real_t Update_psi_LaxWendroff(const LBMState &lbmState) const {
	real_t psi_old  = lbmState[IPSI] ;
	real_t UB_X     = lbmState[IUB_X];
	real_t UB_Y     = lbmState[IUB_Y];
	real_t UB_Z     = 0.0;
	real_t scalar   = UB_X*lbmState[IDPSIDX] + UB_Y*lbmState[IDPSIDY] + UB_Z*lbmState[IDPSIDZ];
	real_t UsUs     = UB_X*UB_X + UB_Y*UB_Y ;
    real_t psi_new  = psi_old - scalar*dt + (dt*dt/2.0)*UsUs*lbmState[ILAPLAPSI];
    return psi_new;
  }
  
  // Update psi_new with psi_old + CT*us.n
  KOKKOS_INLINE_FUNCTION
  real_t Update_psi_CT(const LBMState &lbmState) const {
	real_t psi_old  = lbmState[IPSI] ;
	real_t UB_X     = lbmState[IUB_X];
	real_t UB_Y     = lbmState[IUB_Y];
	real_t UB_Z     = 0.0;
	const real_t norm_psi = sqrt(SQR(lbmState[IDPSIDX]) + SQR(lbmState[IDPSIDY]) +
                             SQR(lbmState[IDPSIDZ])) + NORMAL_EPSILON;
	real_t CT_psi  = counter_term * 4.0 * psi_old * (1.0 - psi_old) / (W_sol * norm_psi);
	real_t scalar  = UB_X*lbmState[IDPSIDX] + UB_Y*lbmState[IDPSIDY] + UB_Z*lbmState[IDPSIDZ];
    real_t psi_new = psi_old - CT_psi*scalar*dt ;
    return psi_new;
  }
  //
  // ===========================================================================================
  // Compute us.grad(psi) for mass balance correction with source term
  KOKKOS_INLINE_FUNCTION
  real_t Sterm_solid_velocity_GradNum (const LBMState &lbmState) const {
	real_t UB_X   = lbmState[IUB_X];
	real_t UB_Y   = lbmState[IUB_Y];
	real_t UB_Z   = 0.0;
    real_t scalar = UB_X*lbmState[IDPSIDX] + UB_Y*lbmState[IDPSIDY] + UB_Z*lbmState[IDPSIDZ];
    real_t Sterm  = scalar ;
    return Sterm;
  }
  
  // Compute CT*us.n for mass balance correction with source term
  KOKKOS_INLINE_FUNCTION
  real_t Sterm_solid_velocity_CT (const LBMState &lbmState) const {
	real_t psi    = lbmState[IPSI ];
	real_t UB_X   = lbmState[IUB_X];
	real_t UB_Y   = lbmState[IUB_Y];
	real_t UB_Z   = 0.0;
	const real_t norm_psi = sqrt(SQR(lbmState[IDPSIDX]) + SQR(lbmState[IDPSIDY]) +
                             SQR(lbmState[IDPSIDZ])) + NORMAL_EPSILON;
	real_t CT_psi  = counter_term * 4.0 * psi * (1.0 - psi) / (W_sol * norm_psi);
    real_t scalar = UB_X*lbmState[IDPSIDX] + UB_Y*lbmState[IDPSIDY] + UB_Z*lbmState[IDPSIDZ];
    real_t Sterm  = CT_psi * scalar ;
    return Sterm;
  }
  // ===========================================================================================
  // Impulsion transfer inside the solid diffuse boundary when the solid velocity is imposed
  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> force_solid_impulsion(const LBMState &lbmState) const {
    real_t penal            = coeff_solid_penalization ;
    const real_t rho_interp = lbmState[ID  ] ;
    const real_t phi        = lbmState[IPHI];
    const real_t phi2       = lbmState[IPHI2];
    const real_t psi        = lbmState[IPSI];
    const real_t nuS        = nu_sol;
    const real_t nu = nu0 * nu1 * nuS / (nu0*nu1*psi + nu1*nuS*phi2 + nu0*nuS*phi);
    real_t kernel     = penal * nu * SQR(psi)*(1.0-psi) / (W_sol*W_sol) ;
    real_t UX         = lbmState[IU   ] ;
    real_t UY         = lbmState[IV   ] ;
    real_t UBX        = lbmState[IUB_X] ;
    real_t UBY        = lbmState[IUB_Y] ;
    RVect<dim> term;
    term[IX] =  solid_phase * kernel * (rho_sol*UBX - rho_interp*UX);
    term[IY] =  solid_phase * kernel * (rho_sol*UBY - rho_interp*UY);
    if (dim == 3) {
		real_t UZ  = lbmState[IW   ] ;
		real_t UBZ = lbmState[IUB_Z] ;
		term[IZ] = solid_phase * kernel * (rho_sol * UBZ - rho_interp*UZ);
    }
    return term;
  }
  // ==============================================================================================================
  // 					                        Models for Navier-Stokes
  // ==============================================================================================================

  // =======================================================
  // relaxation coef for LBM scheme of diffusion equation USED BY MITCHELL ET AL.
  //
  KOKKOS_INLINE_FUNCTION
  real_t tau_NS(const LBMState &lbmState) const {
    const real_t phi = lbmState[IPHI];
    const real_t nu = nu0 * nu1 / (((1.0 - phi) * nu1) + ((phi)*nu0));
    real_t tau = 0.5 + (e2 * nu * dt / SQR(dx));
    return (tau);
  }
  // =======================================================
  // Computation of tauNS by harmonic mean of nu0, nu1 and nuS
  KOKKOS_INLINE_FUNCTION
  real_t tau_NS_harm12S(const LBMState &lbmState) const {
    const real_t phi  = lbmState[IPHI];
    const real_t phi2 = lbmState[IPHI2];
    const real_t psi  = lbmState[IPSI];
    const real_t nuS  = nu_sol;
    const real_t nu = nu0 * nu1 * nuS / (nu0*nu1*psi + nu1*nuS*phi2 + nu0*nuS*phi);
    real_t tau = 0.5 + (e2 * nu * dt / SQR(dx));
    return (tau);
  }
  // =======================================================
  // Harmonic mean for nu with nu0 and nu1
  //
  KOKKOS_INLINE_FUNCTION
  real_t nu_NS_harm12 (const LBMState &lbmState) const {
    const real_t phi  = lbmState[IPHI];
    const real_t nu = nu0 * nu1 / (nu0*phi + nu1*(1.0-phi));
    return (nu);
  }
  // =======================================================
  // Harmonic mean for nu with nu0, nu1 and nuS
  //
  KOKKOS_INLINE_FUNCTION
  real_t nu_NS_harm12S(const LBMState &lbmState) const {
    const real_t phi  = lbmState[IPHI];
    const real_t phi2 = lbmState[IPHI2];
    const real_t psi  = lbmState[IPSI];
    const real_t nuS  = nu_sol;
    const real_t nu = nu0 * nu1 * nuS / (nu0*nu1*psi + nu1*nuS*phi2 + nu0*nuS*phi);
    return (nu);
  }
  // =======================================================
  // Harmonic mean for eta with eta0 and eta1
  //
  KOKKOS_INLINE_FUNCTION
  real_t tau_NS_harm_eta(const LBMState &lbmState) const {
    const real_t phi = lbmState[IPHI];
    const real_t rho = lbmState[ID];
    const real_t eta = rho0*nu0 * rho1*nu1 / (((1.0 - phi) * rho1*nu1) + ((phi)*rho0*nu0));
    real_t tau = 0.5 + (eta/rho) * e2 * dt / SQR(dx);
    return (tau);
  }

  // ===============================================================================================================
  // Pressure term for NS. Correction term because of p* formulation inside the eq dist function
  // ===============================================================================================================
  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> force_P_meth1 (const LBMState &lbmState) const {
	// The numerical gradient of rho is used. 
    RVect<dim> gradrho;
    gradrho[IX] = lbmState[IDRHODX];
    gradrho[IY] = lbmState[IDRHODY];
    if (dim == 3) {
      gradrho[IZ] = lbmState[IDRHODZ];
    }
    const real_t scal = lbmState[IP] / lbmState[ID];
    RVect<dim> term;
    term[IX] = -scal * gradrho[IX];
    term[IY] = -scal * gradrho[IY];
    if (dim == 3) {
      term[IZ] = -scal * gradrho[IZ];
    }
    return term;
  }

  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> force_P_meth2 (const LBMState &lbmState) const {
	// The numerical gradients of phi, phi2 and psi are used
    const real_t scal = lbmState[IP] / lbmState[ID];
    RVect<dim> term;
    term[IX] = -scal * (rho1*lbmState[IDPHIDX] + rho0*lbmState[IDPHI2DX] + rho_sol*lbmState[IDPSIDX]);
    term[IY] = -scal * (rho1*lbmState[IDPHIDY] + rho0*lbmState[IDPHI2DY] + rho_sol*lbmState[IDPSIDY]);
    if (dim == 3) {
      term[IZ] = -scal * (rho1*lbmState[IDPHIDZ] + rho0*lbmState[IDPHI2DZ] + rho_sol*lbmState[IDPSIDZ]);
    }
    return term;
  }
  
  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> force_P_solid (const LBMState &lbmState) const {
	// The numerical gradient of psi is used for solid/liquid interaction
	RVect<dim> ForceP_prime;
	ForceP_prime[IX] = -lbmState[IP] * lbmState[IDPSIDX];
	ForceP_prime[IY] = -lbmState[IP] * lbmState[IDPSIDY];
    if (dim == 3) {
      ForceP_prime[IZ] = -lbmState[IP] * lbmState[IDPSIDZ];
    }
    return ForceP_prime;
  }
  
  // ===============================================================================================================
  // Body term for NS 1: gravity
  //
  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> force_G(const LBMState &lbmState) const {
    RVect<dim> term;
    term[IX] = gx * lbmState[ID];
    term[IY] = gy * lbmState[ID];
    if (dim == 3) {
      term[IZ] = gz * lbmState[ID];
    }
    return term;
  }
  // Body term for NS 2: imposed acceleration inside phase PHI for injection
  //
  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> force_imposed_injct (const LBMState &lbmState) const {
    RVect<dim> term;
    real_t coeff = injector * 0.5*(1.0 + tanh(2.0*(t_inject-time))) ;
    term[IX] = coeff * hx * lbmState[ID] * lbmState[ISPHI];
    term[IY] = coeff * hy * lbmState[ID] * lbmState[ISPHI];
    if (dim == 3) {
      term[IZ] = coeff * hz * lbmState[ID] * lbmState[ISPHI];
    }
    return term;
  }
  // ===============================================================================================================
  // Surface tension liquid/gas for NS
  //
  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> force_TS(const LBMState &lbmState) const {
    const real_t tension =
        sigma * 1.5 / W *
        (g_prime(lbmState[IPHI]) - SQR(W) * lbmState[ILAPLAPHI]);
    RVect<dim> term;
    term[IX] = tension * lbmState[IDPHIDX];
    term[IY] = tension * lbmState[IDPHIDY];
    if (dim == 3) {
      term[IZ] = tension * lbmState[IDPHIDZ];
    }
    return term;
  }

  // ===============================================================================================================
  // Surface tension term number 1 for NS: two-phase with solid-liquid interaction (contact angle)
  //
  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> force_TS1(const LBMState &lbmState) const {
    real_t Gamma1 = Sigma12 + Sigma13 - Sigma23;
    real_t Gamma2 = Sigma12 + Sigma23 - Sigma13;
    real_t Gamma3 = Sigma13 + Sigma23 - Sigma12;
    real_t GammaT = 3.0 / ((1.0 / Gamma1) + (1.0 / Gamma2) + (1.0 / Gamma3));
    const real_t factor = 4.0 * GammaT / W;
    const real_t term1 =
        (1.0 / Gamma2) * ((12.0 * Gamma1 / W) * g2_prime(lbmState[IPHI]) -
                          (12.0 * Gamma2 / W) * g2_prime(lbmState[IPHI2]));
    const real_t term2 =
        (1.0 / Gamma3) * ((12.0 * Gamma1 / W) * g2_prime(lbmState[IPHI]) -
                          (12.0 * Gamma3 / W) * g2_prime(lbmState[IPSI]));
    const real_t mu1_pot = factor * (term1 + term2) -
                           (3.0 / 4.0) * W * Gamma1 * lbmState[ILAPLAPHI];
    RVect<dim> term;
    term[IX] = mu1_pot * lbmState[IDPHIDX];
    term[IY] = mu1_pot * lbmState[IDPHIDY];
    if (dim == 3) {
      term[IZ] = mu1_pot * lbmState[IDPHIDZ];
    }
    return term;
  }
  
  // ===============================================================================================================  
  // Surface tension term number 2 for NS for phi2
  //
  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> force_TS2(const LBMState &lbmState) const {
    real_t Gamma1 = Sigma12 + Sigma13 - Sigma23;
    real_t Gamma2 = Sigma12 + Sigma23 - Sigma13;
    real_t Gamma3 = Sigma13 + Sigma23 - Sigma12;
    real_t GammaT = 3.0 / ((1.0 / Gamma1) + (1.0 / Gamma2) + (1.0 / Gamma3));
    const real_t factor = 4.0 * GammaT / W;
    const real_t term1 =
        (1.0 / Gamma1) * ((12.0 * Gamma2 / W) * g2_prime(lbmState[IPHI2]) -
                          (12.0 * Gamma1 / W) * g2_prime(lbmState[IPHI]));
    const real_t term2 =
        (1.0 / Gamma3) * ((12.0 * Gamma2 / W) * g2_prime(lbmState[IPHI2]) -
                          (12.0 * Gamma3 / W) * g2_prime(lbmState[IPSI]));
    const real_t mu2_pot = factor * (term1 + term2) -
                           (3.0 / 4.0) * W * Gamma2 * lbmState[ILAPLAPHI2];
    RVect<dim> term;
    term[IX] = mu2_pot * lbmState[IDPHI2DX];
    term[IY] = mu2_pot * lbmState[IDPHI2DY];
    if (dim == 3) {
      term[IZ] = mu2_pot * lbmState[IDPHI2DZ];
    }
    return term;
  }
  // ===============================================================================================================  
  // Surface tension term number 3 for NS for psi
  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> force_TS3(const LBMState &lbmState) const {
    real_t Gamma1 = Sigma12 + Sigma13 - Sigma23;
    real_t Gamma2 = Sigma12 + Sigma23 - Sigma13;
    real_t Gamma3 = Sigma13 + Sigma23 - Sigma12;
    real_t GammaT = 3.0 / ((1.0 / Gamma1) + (1.0 / Gamma2) + (1.0 / Gamma3));
    const real_t factor = 4.0 * GammaT / W;
    const real_t term1 =
        (1.0 / Gamma1) * ((12.0 * Gamma3 / W) * g2_prime(lbmState[IPSI]) -
                          (12.0 * Gamma1 / W) * g2_prime(lbmState[IPHI]));
    const real_t term2 =
        (1.0 / Gamma2) * ((12.0 * Gamma3 / W) * g2_prime(lbmState[IPSI]) -
                          (12.0 * Gamma2 / W) * g2_prime(lbmState[IPHI2]));
    const real_t mu3_pot = factor * (term1 + term2) -
                           (3.0 / 4.0) * W * Gamma3 * lbmState[ILAPLAPSI];
    RVect<dim> term;
    term[IX] = mu3_pot * lbmState[IDPSIDX];
    term[IY] = mu3_pot * lbmState[IDPSIDY];
    if (dim == 3) {
      term[IZ] = mu3_pot * lbmState[IDPSIDZ];
    }
    return term;
  }
  // ===============================================================================================================
  // Grad rho
  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> grad_rho(const LBMState &lbmState) const {
    RVect<dim> term;
    real_t phi = lbmState[IPHI];
    real_t c = lbmState[IC];
    real_t drhodphi = (rho1c(lbmState[IC]) - rho0c(lbmState[IC])) *
                      hrho_prime(lbmState[IPHI]);
    real_t drhodc =
        (1 - hrho(phi)) * rho0c_prime(c) + hrho(phi) * rho1c_prime(c);

    term[IX] = drhodphi * lbmState[IDPHIDX] + drhodc * lbmState[IDCDX];
    term[IY] = drhodphi * lbmState[IDPHIDY] + drhodc * lbmState[IDCDY];
    if (dim == 3) {
      term[IZ] = drhodphi * lbmState[IDPHIDZ] + drhodc * lbmState[IDCDZ];
    }
    return term;
  }

  // =======================================================
  // 					Model for composition
  // =======================================================

  KOKKOS_INLINE_FUNCTION
  real_t mu(real_t c, real_t phi) const { return mu_eq + c - c_co(phi); }

  KOKKOS_INLINE_FUNCTION
  real_t tau_C(TagCompo tag, const LBMState &lbmState) const {
    return 0.5 + (e2 * dt / (SQR(dx) * gamma));
  }

  KOKKOS_INLINE_FUNCTION
  real_t M0_C(const LBMState &lbmState) const { return lbmState[IC]; }

  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> M1_C(const LBMState &lbmState) const {
    RVect<dim> term;
    term[IX] = lbmState[IC] * lbmState[IU];
    term[IY] = lbmState[IC] * lbmState[IV];
    if (dim == 3) {
      term[IZ] = lbmState[IC] * lbmState[IW];
    }
    return term;
  }

  KOKKOS_INLINE_FUNCTION
  real_t M2_C(TagCompo tag, const LBMState &lbmState) const {
    return gamma * interpol_D(lbmState[IPHI]) * lbmState[IMU];
  }

  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> S0_C(TagCompo tag,
                                         const LBMState &lbmState) const {
    RVect<dim> term;
    real_t coeff = (D1 - D0) * q_prime(lbmState[IPHI]) * lbmState[IMU];
    term[IX] = coeff * lbmState[IDPHIDX];
    term[IY] = coeff * lbmState[IDPHIDY];
    if (dim == 3) {
      term[IZ] = coeff * lbmState[IDPHIDZ];
    }
    return term;
  }

  // Model without gamma

  KOKKOS_INLINE_FUNCTION
  real_t tau_C(TagSansGamma tag, const LBMState &lbmState) const {
    return 0.5 + (e2 * dt * interpol_D(lbmState[IPHI]) / (SQR(dx)));
  }

  KOKKOS_INLINE_FUNCTION
  real_t M2_C(TagSansGamma tag, const LBMState &lbmState) const {
    return lbmState[IMU];
  }

  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> S0_C(TagSansGamma tag,
                                         const LBMState &lbmState) const {
    RVect<dim> term;
    term[IX] = 0.0;
    term[IY] = 0.0;
    if (dim == 3) {
      term[IZ] = 0.0;
    }
    return term;
  }

  // FUNCTIONS FOR TRT

  // relaxation coef for LBM scheme of phase field equation
  KOKKOS_INLINE_FUNCTION
  real_t tau(const LBMState &lbmState) const {
    const real_t phi = lbmState[IPHI];
    const real_t nu = nu0 * nu1 / (((1.0 - phi) * nu1) + ((phi)*nu0));
    real_t tau = 0.5 + (e2 * nu * dt / SQR(dx));
    return (tau);
  }

  // =======================================================
  // zero order moment for phase field
  KOKKOS_INLINE_FUNCTION
  real_t M0(const LBMState &lbmState) const { return lbmState[IPHI]; }

  // =======================================================
  // first order moment for phase field
  template <int dim>
  KOKKOS_INLINE_FUNCTION RVect<dim> M1(const LBMState &lbmState) const {
    RVect<dim> term;
    term[IX] = lbmState[IPHI] * lbmState[IU];
    term[IY] = lbmState[IPHI] * lbmState[IV];
    if (dim == 3) {
      term[IZ] = lbmState[IPHI] * lbmState[IW];
    }
    return term;
  }

  // =======================================================
  // second order moment for phase field
  KOKKOS_INLINE_FUNCTION
  real_t M2(const LBMState &lbmState) const { return lbmState[IPHI]; }

  // =======================================================
  // Sources term for phase field
  KOKKOS_INLINE_FUNCTION
  real_t S_dwt(const LBMState &lbmState) const {
    real_t source_noct = -(1.0 - counter_term) * Mphi / SQR(W) * 16.0 *
                         lbmState[IPHI] * (1.0 - lbmState[IPHI]) *
                         (1.0 - 2.0 * lbmState[IPHI]);
    return source_noct;
  }

  KOKKOS_INLINE_FUNCTION
  real_t S_stt(const LBMState &lbmState) const {
    real_t S = p_prime(lbmState[IPHI]) * (c0_co - c1_co) *
               (mu(lbmState[IC], lbmState[IPHI]) - mu_eq);
    return -lambda * Mphi / SQR(W) * S;
  }

}; // end struct model
} // namespace PBM_NS_AC_Compo
#endif
