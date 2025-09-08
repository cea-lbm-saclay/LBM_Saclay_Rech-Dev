from math import*

print(" ")
print(" ")
print("###############################################################################")
print("###############################################################################")
print("###                      NUMERICAL VALUES FOR LBM_Saclay                    ###")
print("###                    TEST CASE: RISING BUBBLE (Water-Air)                 ###")
print("###############################################################################")
print("###############################################################################")

################################################################################
## PHYSICAL PROPERTIES (SI UNITS) - ADAPT FOR OTHER FLUIDS
################################################################################
print(" ")
print("#############################################")
print("#           1. INPUT PARAMETERS             #")
print("#############################################")

rho_l = 998.29    # kg/m3  Density water
nu_l  = 1.0034e-6 # m2/s   Kinematic viscosity water
rho_a = 1.204     # kg/m3  Density air
nu_a  = 15.6e-6   # m2/s   Kinematic viscosity air
sigma = 72.8e-3   # N/m    Surface tension Water/Air
g     = 9.81      # m/s2   gravity

print(" ")
print("# PHYSICAL PROPERTIES (SI)")
print("Density   water        :   rho_l = ",rho_l,"kg/m3")
print("Viscosity water        :   nu_l  = ",nu_l ,"m2/s" )
print("Density   air          :   rho_a = ",rho_a,"kg/m3")
print("Viscosity air          :   nu_a  = ",nu_a ,"m2/s" )
print("Surface tension        :   sigma = ",sigma,"N/m"  )
print("Gravity                :   g     = ",g    ,"m/s2" )


#####################################################
## DERIVED PROPERTIES
#####################################################
print(" ")
print("# DERIVED PROPERTIES")
DeltaRho = rho_l - rho_a
Density_ratio = rho_l / rho_a
Viscosity_ratio = nu_l/nu_a

Visc_Dyn_l = rho_l*nu_l
Visc_Dyn_a = rho_a*nu_a
Visc_Dyn_ratio = Visc_Dyn_l / Visc_Dyn_a

print("Density   ratio        :   rho_l/rho_a     = ",Density_ratio)
print("Viscosity ratio        :   nu_l / nu_a     = ",Viscosity_ratio, " = 1/",nu_a/nu_l)
print("DeltaRho               :   rho_l-rho_a     = ",DeltaRho)
print("Dynamic viscosity water:   Visc_Dyn_l      = ",Visc_Dyn_l)
print("Dynamic viscosity air  :   Visc_Dyn_a      = ",Visc_Dyn_a)
print("Dynamic viscosity ratio:   Visc_l / Visc_a = ",Visc_Dyn_ratio)

#####################################################
## HYPOTHESES FOR SPLASH - ADAPT FOR OTHER TEST CASE
#####################################################
print(" ")
print("# HYPOTHESES FOR RISING BUBBLE (ADAPT FOR OTHER TEST CASE)")
# Viscosity of reference is nu_l (liquid)
nu=nu_l
R = 0.001          # m (droplet radius)
D = 2*R            # m (droplet diameter)
# Hypothesis: Uref is given by sqrt(g*D)
print("Hypothesis: Uref=sqrt(g*D) where g=9.81m/s2 and D the droplet diameter")
Uref = sqrt(g*D)

print("For D = ",D   , "m then:")
print("Uref  = sqrt(gD)                   = ",Uref, "m/s")
#####################################################
## COMPUTATION OF REYNOLDS, BOND AND MORTON NUMBERS
#####################################################
print(" ")
print("# COMPUTATION OF REYNOLDS, BOND, AND MORTON NUMBERS")
# Re, Bo and Mo are derived
Re = Uref*D/nu
Bo = g*DeltaRho*D**2/sigma
Mo = g*DeltaRho * Visc_Dyn_l**4 / (rho_l**2 * sigma**3)
print("Re    = Uref*D/nu_l                              = ",Re)
print("Bo    = g*DeltaRho * D**2  / sigma               = ",Bo)
print("Mo    = g*DeltaRho * eta^4 / (rho_l^2 * sigma^3) = ",Mo)

################################################################################
## PROPOSITION DE VALEURS POUR LBM_Saclay
################################################################################
print(" ")
print("###############################################################################")
print("###############################################################################")
print(" ")
print("#############################################")
print("#           2. ADIM PARAMETERS              #")
print("#############################################")
## VALEURS ADIMENSIONNEES FIXEES

print(" ")
print("# We set Rstar, rho_ref, tau_l_star")

Rstar      = 37.5
#rho_ref    = rho_l/100.0
#print("# The reference density is rho_l/100")
rho_ref    = rho_l
print("# The reference density is rho_l")
tau_l_star = 0.51

L     = (400./Rstar)*R # m (domain width)
Dstar = 2*Rstar
print("Rayon adim             : Rstar      = ", Rstar)
print("Collision rate of water: tau_l_star = ", tau_l_star)

## DERIVED DIMENSIONLESS VALUES
print(" ")
print("# DERIVED DIMENSIONLESS VALUES")
# Length
dx            = R/Rstar
Lstar         = L/dx
# Densités
rho_l_star    = rho_l / rho_ref
rho_a_star    = rho_a / rho_ref
DeltaRho_star = rho_l_star - rho_a_star
nu_l_star     = (1./3.)*(tau_l_star-0.5)

# Sigma_star derived with Bo & Mo
Sigma_star    = sqrt((Bo*(rho_l_star*nu_l_star)**4)/(Mo*DeltaRho_star*Dstar**2*rho_l_star))
# gstar derived with Re
gstar         = Sigma_star*Bo/(DeltaRho_star * Dstar**2)

dt            = sqrt(gstar*dx/g)
tau_a_star    = 3.0*nu_a*dt/(dx**2)+0.5
nu_a_star     = (1./3.)*(tau_a_star-0.5)
Uref_star     = sqrt(gstar*Dstar)

print("dx            =",dx        )
print("Lstar         =",Lstar     )
print("rho_l_star    =",rho_l_star)
print("rho_a_star    =",rho_a_star)
print("DeltaRho_star =",DeltaRho_star)
print("nu_l_star     =",nu_l_star , "(derived from tau_l_star)")
print("Sigma_star    =",Sigma_star, "(derived from Bo & Mo)")
print("gstar         =",gstar     , "(derived with Bo & Sigma_star)")

print("dt            =",dt        )
print("tau_a_star    =",tau_a_star)
print("Uref_star     =",Uref_star )

################################################################################
## VERIFICATIONS OF DIMENSIONLESS VALUES
################################################################################
print(" ")
print("###############################################################################")
print("###############################################################################")
print(" ")
print("#############################################")
print("#           3. VERIFICATION                 #")
print("#############################################")
print(" ")
print("# VERIFICATIONS OF DIMENSIONLESS VALUES WITH 'star', dx, dt et rho_ref")
print(" ")
print("Use of")
print("dx      = ",dx)
print("dt      = ",dt)
print("rho_ref = ",rho_ref)
print(" ")
print("SIMILARITY OF DIMENSIONLESS PARAMETERS")
L_verif     = Lstar      * dx
g_verif     = gstar      * dx/(dt**2)
R_verif     = Rstar      * dx
rho_l_verif = rho_l_star * rho_ref
rho_a_verif = rho_a_star * rho_ref
nu_l_verif  = nu_l_star  * dx**2/dt
nu_a_verif  = nu_a_star  * dx**2/dt
Sigma_verif = Sigma_star * rho_ref*dx**3/(dt**2)

Uref_verif  = Uref_star  * dx/dt

print("L_verif     (m)     = Lstar * dx                         =", L_verif)
print("g_verif     (m/s2)  = gstar * dx / dt2                   =", g_verif)
print("R_verif     (m)     = Rstar * dx                         =", R_verif)
print("rho_l_verif (kg/m3) = rho_l_star * rho_ref               =", rho_l_verif)
print("rho_a_verif (kg/m3) = rho_a_star * rho_ref               =", rho_a_verif)
print("nu_l_verif  (m2/s)  = nu_l_star  * dx2 / dt              =", nu_l_verif)
print("nu_a_verif  (m2/s)  = nu_a_star  * dx2 / dt              =", nu_a_verif)
print("Sigma_verif (N/m)   = Sigma_star * rho_ref * dx3 / (dt2) =", Sigma_verif)
print("Uref_verif  (m/s)   = Uref_star  * dx / dt               =", Uref_verif)

################################################################################
## PROPOSITIONS DE VALEURS POUR LBM_Saclay
################################################################################
print(" ")
print("###############################################################################")
print("###############################################################################")
print("# EXAMPLE OF INPUT VALUES FOR LBM_Saclay")

dt_star = 1.0
dx_star = 1.0

xmin = 0.0
nx   = int(Lstar)
xmax = nx*dx_star
ymin = 0.0
ny   = nx*2
ymax = ny*dx_star

print("[run]")
print("dt    =",dt_star)
print("[mesh]")
print("nx    =",nx)
print("ny    =",ny)
print("xmin  =",xmin)
print("xmax  =",xmax)
print("ymin  =",ymin)
print("ymax  =",ymax)
print("[params]")
print("rho0  =",rho_a_star)
print("rho1  =",rho_l_star)
print("nu0   =",nu_a_star)
print("nu1   =",nu_l_star)
print("gy    =",-gstar)
print("sigma =",Sigma_star)
print("[init]")
print("rayon =",Rstar)


print(" ")
print("###############################################################################")
print("###############################################################################")
print("###########################        END         ################################")
print("###############################################################################")
print("###############################################################################")
