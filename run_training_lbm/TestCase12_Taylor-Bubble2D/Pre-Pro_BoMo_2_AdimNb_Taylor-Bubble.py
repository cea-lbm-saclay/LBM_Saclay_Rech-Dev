from math import*
print(" ")
print(" ")
print("###############################################################################")
print("###############################################################################")
print("###                            BOND & MORTON TO                             ###")
print("###                        DIMENSIONLESS PARAMETERS                         ###")
print("###                       TEST CASE: TAYLOR's BUBBLE                        ###")
print("###############################################################################")
print("###############################################################################")

print(" ")
print("# COMMENT")
print("# COMPUTATION OF eta_l_star and sigma_star with Bo & Mo")
print(" ")

################################################################
## INPUT PARAMETERS
################################################################

#Paramètres de la simulation
dt     = 1
Nx     = 150
L_star = 150
L      = L_star
dx = L/Nx

print("dx = ",dx)


W     = 8.0
Mphi  = 0.01

rho0  = 0.0013440860215053765
rho1  = 1.0
nu0   = 0.006666666666666672
D     = L
g     = 1e-6
# DERIVED PROPERTIES
DeltaRho = rho1-rho0
R        = 0.75*D/2.0

print(" ")
print("# INPUT PARAMETERS")
print("rho0     = ", rho0   )
print("rho1     = ", rho1   )
print("nu0      = ", nu0    )
print("g        = ", g      )
print("D        = ", D      )
print("# DERIVED PARAMETERS")
print("R        = ", R      )
print("DeltaRho = ", DeltaRho)

###############################################################
## TARGET ADIM NB Bo & Mo (comment/decomment the case)
################################################################

# TAYLOR bubble CASE 1
#Bo=137.32910156545728
#Mo=9.374999999798115

# TAYLOR bubble CASE 4
#Bo=100.0
#Mo=0.5

# TAYLOR bubble CASE 5
Bo=100.0
Mo=10.0

# TAYLOR bubble CASE 6
#Bo=100.0
#Mo=0.1

# TAYLOR bubble CASE 7
#Bo=100.0
#Mo=1.0

# TAYLOR bubble CASE 8
#Bo=100.0
#Mo=0.05

# TAYLOR bubble CASE 9
#Bo=100.0
#Mo=0.015

print(" ")
print("# TARGET ADIM Bo & Mo")
print("Bo    = ", Bo    )
print("Mo    = ", Mo    )
###############################################################
## COMPUTATION OF sigma and eta_l
################################################################

sigma = g*DeltaRho*D**2/Bo
eta_l = (Mo*DeltaRho*sigma**3/g)**(1./4)

print(" ")
print("# COMPUTED g_star & sigma_star")
print("sigma =",sigma)
print("eta_l =",eta_l  )

print(" ")
print("###############################################################################")
print("###############################################################################")
print("###########################        END         ################################")
print("###############################################################################")
print("###############################################################################")

