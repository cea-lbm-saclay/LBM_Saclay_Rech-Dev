from math import*
print(" ")
print(" ")
print("###############################################################################")
print("###############################################################################")
print("###                            BOND & MORTON TO                             ###")
print("###                        DIMENSIONLESS PARAMETERS                         ###")
print("###                        TEST CASE: RISING BUBBLE                         ###")
print("###############################################################################")
print("###############################################################################")

print(" ")
print("# COMMENT")
print("# COMPUTATION OF g_star and sigma_star with Bo & Mo")
print(" ")

################################################################
## INPUT PARAMETERS
################################################################

#Paramètres de la simulation
dt     = 1
Nx     = 400
L_star = 400
L      = L_star
dx = L/Nx

print("dx = ",dx)


W     = 8.0
Mphi  = 0.0054481546878700492

rho0  = 1.2060623666469664e-3
rho1  = 1.0
nu0   = 0.051839436555692744
nu1   = 0.003333333333333336
R     = 37.5
# DERIVED PROPERTIES
DeltaRho = rho1-rho0
eta1     = rho1*nu1
D        = 2.0*R

print(" ")
print("# INPUT PARAMETERS")
print("rho0     = ", rho0   )
print("rho1     = ", rho1   )
print("nu0      = ", nu0    )
print("nu1      = ", nu1    )
print("R        = ", R      )
print("# DERIVED PARAMETERS")
print("D        = ", D      )
print("DeltaRho = ", DeltaRho)
print("eta1     = ", eta1)

###############################################################
## TARGET ADIM NB Bo & Mo (comment/decomment the case)
################################################################

# CASE A1
#Bo    = 17.7
#Mo    = 711

# CASE A2
Bo    = 32.2
Mo    = 8.2e-4

# CASE A3
#Bo    = 243
#Mo    = 266

# CASE A4
#Bo    = 115
#Mo    = 4.63e-3

# CASE A5
#Bo    = 339
#Mo    = 43.1

print(" ")
print("# TARGET ADIM Bo & Mo")
print("Bo    = ", Bo    )
print("Mo    = ", Mo    )
###############################################################
## COMPUTATION OF g_star and sigma_star
################################################################

#sigma = sqrt((Bo*(rho1*nu1)**4)/(Mo*rho1**2*D**2))
factr = eta1**2 / (rho1*D)
sigma = factr * sqrt(Bo/Mo)

gy    = sigma*Bo/(DeltaRho * D**2)

print(" ")
print("# COMPUTED g_star & sigma_star")
print("sigma =",sigma)
print("gy    =",gy  )

print(" ")
print("###############################################################################")
print("###############################################################################")
print("###########################        END         ################################")
print("###############################################################################")
print("###############################################################################")

