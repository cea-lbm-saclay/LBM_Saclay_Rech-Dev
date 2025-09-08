import numpy as np
import matplotlib.pyplot as plt
import csv

L = 1e-3
nu = 1e-6
g=10
Re = 1250
u_max = g * L**2 / (8*nu)

dx = 1e-5
tau = 0.55
dx_star = 1
dt_star = 1

L_star = dx_star *L/dx
nu_star = 1/3 * (tau - 0.5)
dt = nu_star / nu * dx**2
g_star = g* dt**2 /dx

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

T_f_star = 1000000
T_f = dt * T_f_star/ dt_star
print("T_f = ", T_f, "s")


u_star=np.array([])
x_star=np.array([])

with open('Poiseuille_Water.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if row[0]!= 'laplaphi':
            x_star = np.append(x_star, float(row[32]))
            u_star = np.append(u_star, float(row[10]))

x = dx*x_star
u = dx/dt * u_star
u_analytique = u_max- u_max*4*(x*x)/(L*L)

fig,ax = plt.subplots()
fig.set_size_inches(12,8, forward = True)
l1, = ax.plot( u,x, color= 'r')
l2, = ax.plot(u_analytique,x, linestyle= "None",marker = 'o', markerfacecolor = 'blue', markeredgecolor='blue', markevery= 2)

ax.legend( (l1,l2), ('simulation','analytique'), loc= 'upper right', shadow = True)

ax.set_xlabel('$y$')
ax.set_ylabel('$u_x(y)$')
plt.show()

