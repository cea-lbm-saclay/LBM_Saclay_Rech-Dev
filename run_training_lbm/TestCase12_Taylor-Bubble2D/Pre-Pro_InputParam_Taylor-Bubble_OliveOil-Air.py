from math import*

print(" ")
print(" ")
print("###############################################################################")
print("###############################################################################")
print("###                      NUMERICAL VALUES FOR LBM_Saclay                    ###")
print("###                TEST CASE: TAYLOR BUBBLE (Olive oil - Air)               ###")
print("###############################################################################")
print("###############################################################################")

################################################################################
## PHYSICAL PROPERTIES (SI UNITS) - ADAPT FOR OTHER FLUIDS
################################################################################
print(" ")
print("#############################################")
print("#           1. INPUT PARAMETERS             #")
print("#############################################")

rho_l = 911.4      # kg/m3  Density olive oil
eta_l = 0.08399988 # Pa.s   Dynamic viscosity olive oil
rho_a = 1.225      # kg/m3  Density air (from ref [1] in Readme)
eta_a = 1.983e-5   # Pa.s   Dynamic viscosity air (from ref [1] in Readme)
sigma = 32e-3      # N/m    Surface tension Olive oil/Air
g     = 9.81       # m/s2   gravity

print(" ")
print("# PHYSICAL PROPERTIES (SI)")
print("Density   olive oil    :   rho_l = ",rho_l,"kg/m3")
print("Viscosity olive oil    :   eta_l = ",eta_l,"Pa.s" )
print("Density   air          :   rho_a = ",rho_a,"kg/m3")
print("Viscosity air          :   eta_a = ",eta_a,"Pa.s" )
print("Surface tension        :   sigma = ",sigma,"N/m"  )
print("Gravity                :   g     = ",g    ,"m/s2" )


#####################################################
## DERIVED PROPERTIES
#####################################################
print(" ")
print("# DERIVED PROPERTIES")
DeltaRho = rho_l - rho_a
Density_ratio = rho_l / rho_a
Viscosity_ratio = eta_l/eta_a

nu_l = eta_l/rho_l
nu_a = eta_a/rho_a
nu_ratio = nu_l/nu_a

print("Density   ratio              :   rho_l/rho_a   = ",Density_ratio)
print("Viscosity ratio              :   eta_l / eta_a = ",Viscosity_ratio)
print("DeltaRho                     :   rho_l-rho_a   = ",DeltaRho)
print("Kinematic viscosity olive oil:   Kin visc_l    = ",nu_l)
print("Kinematic viscosity air      :   Kin visc_a    = ",nu_a)
print("Kinematic viscosity ratio    :   nu_l / nu_    = ",nu_ratio)

#####################################################
## HYPOTHESES FOR SPLASH - ADAPT FOR OTHER TEST CASE
#####################################################
print(" ")
print("# HYPOTHESES FOR TAYLOR's BUBBLE (ADAPT FOR OTHER TEST CASE)")
# Viscosity of reference is nu_l (liquid)
nu=nu_l

D = 0.0019         # m (tube diameter)
R = 0.75*D/2       # m (Taylor's bubble radius)
# Hypothesis: Uref is given by sqrt(g*D)
print("Hypothesis: Uref=sqrt(g*D) where g=9.81m/s2 and D is the tube diameter")
Uref = sqrt(g*D)

print("For D = ",D   , "m then:")
print("Uref  = sqrt(gD)                   = ",Uref, "m/s")
#####################################################
## COMPUTATION OF REYNOLDS, BOND AND MORTON NUMBERS
#####################################################
print(" ")
print("# COMPUTATION OF REYNOLDS, BOND, AND MORTON NUMBERS")
print("# Definitions of Bo and Mo are from ref [1]")
# Re, Bo and Mo are derived
Re = Uref*D/nu
Bo = g*DeltaRho*D**2/sigma
Mo = g * eta_l**4 / (DeltaRho * sigma**3)
print("Re    = Uref*D/nu_l                      = ",Re)
print("Bo    = g*DeltaRho * D**2  / sigma       = ",Bo)
print("Mo    = g*eta_l^4 / (DeltaRho * sigma^3) = ",Mo)

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
print("# We set Dstar, rho_ref, tau_a_star")
print("# The reference density is rho_l")
Dstar      = 150
rho_ref    = rho_l
tau_a_star = 0.52

L     = (150./Dstar)*D # m (domain width)
print("Tube diameter adim       : Dstar      = ", Dstar)
print("Reference density        : rho_ref    = rho_l = ",rho_ref)
print("Set collision rate of air: tau_l_star = ", tau_a_star)

## DERIVED DIMENSIONLESS VALUES
print(" ")
print("# DERIVED DIMENSIONLESS VALUES")
# Length
dx            = D/Dstar
Lstar         = L/dx
# Densités
rho_l_star    = rho_l / rho_ref
rho_a_star    = rho_a / rho_ref
DeltaRho_star = rho_l_star - rho_a_star
nu_a_star     = (1./3.)*(tau_a_star-0.5)

dt = (nu_a_star/nu_a)*dx**2

nu_l_star = nu_l*dt/(dx*dx)
tau_l_star    = 3.0*nu_l*dt/(dx**2)+0.5

# Sigma_star derived with Bo & Mo
Sigma_star    = sqrt((Bo*(rho_l_star*nu_l_star)**4)/(Mo*DeltaRho_star*Dstar**2*rho_l_star))
# gstar derived with Re
gstar         = Sigma_star*Bo/(DeltaRho_star * Dstar**2)
Rstar         = 0.75*Dstar/2.0
Uref_star     = sqrt(gstar*Dstar)

print("dx            =",dx        )
print("Lstar         =",Lstar     )
print("rho_l_star    =",rho_l_star)
print("rho_a_star    =",rho_a_star)
print("DeltaRho_star =",DeltaRho_star)
print("nu_a_star     =",nu_a_star , "(derived from tau_a_star)")
print("nu_l_star     =",nu_l_star , "(derived from tau_l_star)")
print("Sigma_star    =",Sigma_star, "(derived from Bo & Mo)")
print("gstar         =",gstar     , "(derived with Bo & Sigma_star)")

print("dt            =",dt        )
print("tau_a_star    =",tau_a_star)
print("tau_l_star    =",tau_l_star)
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
D_verif     = Dstar      * dx
rho_l_verif = rho_l_star * rho_ref
rho_a_verif = rho_a_star * rho_ref
nu_l_verif  = nu_l_star  * dx**2/dt
nu_a_verif  = nu_a_star  * dx**2/dt
Sigma_verif = Sigma_star * rho_ref*dx**3/(dt**2)

Uref_verif  = Uref_star  * dx/dt

print("L_verif     (m)     = Lstar * dx                         =", L_verif)
print("g_verif     (m/s2)  = gstar * dx / dt2                   =", g_verif)
print("D_verif     (m)     = Rstar * dx                         =", D_verif)
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
