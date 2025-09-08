from math import*

print(" ")
print(" ")
print("###############################################################################")
print("###############################################################################")
print("###                      NUMERICAL VALUES FOR LBM_Saclay                    ###")
print("###                        TEST CASE: RAYLEIGH-TAYLOR                       ###")
print("###############################################################################")
print("###############################################################################")
# Paramètres numériques de la simulation
L = 128
dt = 1
dx = 1
Nx = 128
dx=L/Nx

# Paramètres physiques de la simulation
print(" ")
print("# INPUT PARAMETERS (ADIM)")
rhoH     = 3
rhoL     = 1
DeltaRho = rhoH - rhoL
At       = (rhoH-rhoL)/(rhoH+rhoL)
print("rhoH = ",rhoH)
print("rhoL = ",rhoL)
print("At   = ",At)
print("L    = ",L)
print("Nx   = ",Nx)

# TARGET ADIM NB
print("# TARGET ADIM NB")
Re = 3000
Pe = 1000
Ca = 0.26
print("Re = ",Re)
print("Pe = ",Pe)
print("Ca = ",Ca)

# Fixation du temps caractéristique
Tcar = 16000 * dt
print("# CHARACTERISTIC TIME OF SIMULATION")
print("Tcar = ",Tcar)
print(" ")
print("# DERIVATION OF INPUT PARAMETERS")
g     = L/(Tcar**2)/At
# Calculs des grandeurs manquante
W     = 4 * dx
nu    = (L*sqrt(L*g))/Re
sigma = (rhoH*nu*sqrt(L*g))/Ca
M     = (L*sqrt(L*g))/Pe
lc    = sqrt(sigma/(g*DeltaRho))
# OUTPUT
print("g     = ",g, "(derived from L, Tcar and At)")
print("M     = ",M,"(derived from Pe)")
print("nu    = ",nu, "(derived from Re)")
print("sigma = ",sigma,"(derived from Ca)")
print("lc    = ",lc,"Capillary length")
print("W     = ",W)

print(" ")
print("###############################################################################")
print("###############################################################################")
print("###########################        END         ################################")
print("###############################################################################")
print("###############################################################################")

