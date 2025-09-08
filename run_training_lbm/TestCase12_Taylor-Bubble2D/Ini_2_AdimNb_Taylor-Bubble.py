from math import*
print(" ")
print(" ")
print("###############################################################################")
print("###############################################################################")
print("###                       .ini FILE OF LBM_Saclay TO                        ###")
print("###                        DIMENSIONLESS PARAMETERS                         ###")
print("###                        TEST CASE: RISING BUBBLE                         ###")
print("###############################################################################")
print("###############################################################################")

print(" ")
print("# COMMENT")
print("# VERIFICATION OF ADIM NB CALCULATED WITH INPUT PARAMETERS OF LBM_Saclay")
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
Mphi  = 0.0054481546878700492

rho0  = 0.001344086
rho1  = 1.0
nu0   = 0.01666666666666668
nu1   = 0.12233786033898852
sigma = 0.0003159472821124211
gy    = 1.406100062160259e-6

D_star = 150

print(" ")
print("# INPUT PARAMETERS")
print("W     = ", W     )
print("Mphi  = ", Mphi  )
print("rho0  = ", rho0  )
print("rho1  = ", rho1  )
print("nu0   = ", nu0   )
print("nu1   = ", nu1   )
print("g     = ", gy    )
print("sigma = ", sigma )
print("D     = ", D_star)

###############################################################
## COMPUTATION OF DIMENSIONLESS PARAMETERS
################################################################
eta1 = rho1*nu1
DeltaRho = rho1-rho0

Ucar = sqrt(gy*D_star)
Re   = D_star*Ucar/nu1
Bo   = gy * DeltaRho * D_star**2 / sigma
Mo   = gy * eta1**4 / (DeltaRho*sigma**3)
Pe   = D_star*Ucar/Mphi
#Uverif = nu1*Re / D_star
# Caracteristic time
t_c  = D_star / Ucar

print(" ")
print("# DERIVED DIMENSIONLESS NUMBERS")
print("Ucar   = sqrt(g*D)                        =",Ucar)
print("Re     = Ucar*R/nu_l                      =",Re  )
print("Bo     = DeltaRho*g*D**2 / sigma          =",Bo  )
print("Mo     = g* eta1**4 / (DeltaRho*sigma**3) =",Mo  )
print("Pe     = D*Ucar/Mphi                      =",Pe  )
print("t_c    = D/Ucar                           =",t_c )

print(" ")
print("###############################################################################")
print("###############################################################################")
print("###########################        END         ################################")
print("###############################################################################")
print("###############################################################################")

