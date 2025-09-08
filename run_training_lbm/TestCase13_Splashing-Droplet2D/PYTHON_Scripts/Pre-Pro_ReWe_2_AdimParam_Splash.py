from math import*
print(" ")
print(" ")
print("###############################################################################")
print("###############################################################################")
print("###                          REYNOLDS & WEBER TO                            ###")
print("###                        DIMENSIONLESS PARAMETERS                         ###")
print("###                        TEST CASE: RISING BUBBLE                         ###")
print("###############################################################################")
print("###############################################################################")

print(" ")
print("# COMMENT")
print("# COMPUTATION OF nu1 and sigma_star with Re & We")
print(" ")

################################################################
## INPUT PARAMETERS
################################################################

#Paramètres de la simulation
dt     = 1
Nx     = 1024
L_star = 1024
L      = L_star
dx = L/Nx

print("dx = ",dx)


W     = 8.0
Mphi  = 0.0054481546878700492

rho0  = 0.12060623666469664
rho1  = 100.0
nu0   = 0.010364759816623473
R     = 30
# DERIVED PROPERTIES
DeltaRho = rho1-rho0
D        = 2.0*R

print(" ")
print("# INPUT PARAMETERS")
print("rho0     = ", rho0   )
print("rho1     = ", rho1   )
print("nu0      = ", nu0    )
print("R        = ", R      )
print("# DERIVED PARAMETERS")
print("D        = ", D      )
print("DeltaRho = ", DeltaRho)

###############################################################
## TARGET ADIM NB Bo & Mo (comment/decomment the case)
################################################################

Re=500
We=8000
Ucar  = 0.03

print(" ")
print("# TARGET ADIM Re & We")
print("Re    = ", Re         )
print("We    = ", We         )
print("# Target velocity"    )
print("Ucar  = ", Ucar       )
###############################################################
## COMPUTATION OF nu_l and sigma_star
################################################################
nu1   = Ucar*D/Re
sigma = rho1*Ucar*Ucar*D/We

print(" ")
print("# COMPUTED g_star & sigma_star")
print("sigma =",sigma)
print("nu_l    =",nu1)

print(" ")
print("###############################################################################")
print("###############################################################################")
print("###########################        END         ################################")
print("###############################################################################")
print("###############################################################################")

