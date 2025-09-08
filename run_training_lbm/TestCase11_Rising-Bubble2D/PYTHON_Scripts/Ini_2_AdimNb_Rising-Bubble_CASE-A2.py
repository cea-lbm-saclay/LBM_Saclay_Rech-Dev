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
Nx     = 256
L_star = 256
L      = L_star
dx = L/Nx

print("dx = ",dx)


W     = 8.0
Mphi  = 0.05

rho0  = 0.0012060623666469664
rho1  = 1.0
nu0   = 0.051839436555692744
nu1   = 0.003333333333333336
sigma = 2.9357376111218993e-5
gy    = 1.6825759773987011e-7

R_star   = 37.5
D_star   = 2*R_star
DeltaRho = rho1 - rho0

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
print("R     = ", R_star)

###############################################################
## COMPUTATION OF DIMENSIONLESS PARAMETERS
################################################################
Ucar = sqrt(gy*2*D_star)
Re   = D_star*Ucar/nu1
Bo   = gy*DeltaRho*D_star**2/sigma
Mo   = gy*DeltaRho*(rho1*nu1)**4 / (rho1**2*sigma**3)
Pe   = D_star*Ucar/Mphi
#Uverif = nu1*Re / D_star

# Caracteristic time
t_c  = R_star / Ucar

print(" ")
print("# DERIVED DIMENSIONLESS NUMBERS")
print("Ucar   = sqrt(g*R)                       =",Ucar)
print("Re     = Ucar*R/nu_l                     =",Re  )
print("Bo     = (rho_w-rho_a)*g*R**2/sigma      =",Bo  )
print("Mo     = g*DeltaRho*(rho1*nu1)**4/(rho1**2*sigma**3) =",Mo  )
print("Pe     = R*Ucar/Mphi                     =",Pe  )
print("t_c    = R/Ucar                          =",t_c )
#print("Uverif = Re*nu_l/D             = ",Uverif)

print(" ")
print("###############################################################################")
print("###############################################################################")
print("###########################        END         ################################")
print("###############################################################################")
print("###############################################################################")

