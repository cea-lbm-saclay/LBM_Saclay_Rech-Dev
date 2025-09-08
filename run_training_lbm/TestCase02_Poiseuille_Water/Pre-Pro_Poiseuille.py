import numpy as np
import matplotlib.pyplot as plt
import csv

print(" ")
print(" ")
print("###############################################################################")
print("###############################################################################")
print("###                      NUMERICAL VALUES FOR LBM_Saclay                    ###")
print("###                     TEST CASE: POISEUILLE (Water-Air)                   ###")
print("###############################################################################")
print("###############################################################################")


################################################################################
## PHYSICAL PROPERTIES (SI UNITS) - ADAPT FOR OTHER FLUIDS
################################################################################
print(" ")
print("#########################################################")
print("#  1. PHYSICAL PARAMETERS (SI UNITS)                    #")
print("#########################################################")
L = 1e-3
nu = 1e-6
g=10
Re = 1250
u_max = g * L**2 / (8*nu)

print("L     =",L,"m")
print("nu    =",nu,"m/s")
print("g     =",g,"m/s2")
print("Re    =",Re)
print("u_max =",u_max,"m/s")

print(" ")
print("#########################################################")
print("#  2. CHOICE OF IMPOSED PARAMETERS                      #")
print("#########################################################")
dx = 1e-5
tau = 0.55
dx_star = 1
dt_star = 1
print("dx      =",dx     )
print("tau     =",tau    )
print("dx_star =",dx_star)
print("dt_star =",dt_star)

print(" ")
print("#########################################################")
print("#  3. DERIVATION OF DIMENSIONLESS SET OF PARAMETERS     #")
print("#########################################################")

L_star  = dx_star *L/dx
nu_star = 1/3 * (tau - 0.5)
dt      = nu_star / nu * dx**2
g_star  = g* dt**2 /dx

u_max_star = nu_star * Re / L_star

print("dx         = ", dx, " m")
print("dt         = ", dt, " s")
print("dx_star    = ", dx_star)
print("dt_star    = ", dt_star)
print("tau        = ", tau)
print("L_star     = ", L_star)
print("nu_star    = ", nu_star)
print("g_star     = ", g_star)
print("u_max_star = ", u_max_star )

print(" ")
print("###############################################################################")
print("###############################################################################")
print("###########################        END         ################################")
print("###############################################################################")
print("###############################################################################")


