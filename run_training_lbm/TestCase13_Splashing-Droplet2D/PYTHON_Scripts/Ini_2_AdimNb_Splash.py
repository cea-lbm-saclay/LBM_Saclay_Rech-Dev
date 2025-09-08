from math import*
print(" ")
print(" ")
print("###############################################################################")
print("###############################################################################")
print("###                       .ini FILE OF LBM_Saclay TO                        ###")
print("###                        DIMENSIONLESS PARAMETERS                         ###")
print("###                           TEST CASE: SPLASH                             ###")
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
Nx     = 1024
L_star = 1024
L      = L_star
dx = L/Nx

print("dx = ",dx)
W=8.0
Mphi=0.0054481546878700492

rho0=0.12060623666469664
rho1=100.0
nu0=0.010364759816623473
nu1=0.0006666666666666672
sigma=2.1375e-4

gy=1.581e-6


R_star = 30.
D_star = 2*R_star

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
print("D     = ", D_star)

###############################################################
## COMPUTATION OF DIMENSIONLESS PARAMETERS
################################################################
Ucar = 3.3e-2
Re   = D_star*Ucar/nu1
We   = rho1*Ucar*Ucar*D_star/sigma
Pe   = D_star*Ucar/Mphi
#Uverif = nu1*Re / D_star

# Caracteristic time
t_c  = D_star / Ucar

print(" ")
print("# DERIVED DIMENSIONLESS NUMBERS")
print("Ucar   = sqrt(g*D)             = ",Ucar)
print("Re     = Ucar*D/nu_l           = ",Re  )
print("We     = rho_l*Ucar**2*D/sigma = ",We  )
print("Pe     = D*Ucar/Mphi           = ",Pe  )
print("t_c    = D/Ucar                = ",t_c )
#print("Uverif = Re*nu_l/D             = ",Uverif)

print(" ")
print("###############################################################################")
print("###############################################################################")
print("###########################        END         ################################")
print("###############################################################################")
print("###############################################################################")

