import numpy as np
import matplotlib.pyplot as plt
import csv

u_c = 5e-5
h = 1.28
eta_B = 0.00166666666
print("xmin = ", -h)
print("xmax = ", h)
print("nu0  = ", eta_B)

eta_A = 0.166666666
G= u_c * (eta_A + eta_B) / h**2
print("nu1 = ", eta_A)
print("gy  = ", G)

x_list=np.array([])
u_y = np.array([])
with open('Profil_Double-Poiseuille_MRT.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if row[0]!= 'laplaphi':
            x_list = np.append(x_list, float(row[21]))
            u_y = np.append(u_y, float(row[8]))
                    
u_th = np.zeros(len(x_list))
for i,x in enumerate(x_list):
    if x >=0:
        u_th[i] = G * h**2 / (2*eta_A) * (-(x/h)**2 - x/h * (eta_A - eta_B)/(eta_A + eta_B) + 2 * eta_A / (eta_A + eta_B))
    else: 
        u_th[i] = G * h**2 / (2*eta_B) * (-(x/h)**2 - x/h * (eta_A - eta_B)/(eta_A + eta_B) + 2 * eta_B / (eta_A + eta_B))

x_list = x_list /h

fig,ax = plt.subplots()
fig.set_size_inches(12,8, forward = True)
l1, = ax.plot(u_th,x_list, color = 'r')
l2, = ax.plot(u_y,x_list, linestyle= 'None',marker = 'o', markerfacecolor = 'blue', markeredgecolor='blue', markevery= 4)

ax.legend( (l1,l2), ('analytique','simulation'), loc= 'upper right', shadow = True)
ax.set_title("Profil d'un écoulement de double Poiseuille pour $ \eta_A / \eta_B = 100$" )
ax.set_ylabel('$x$')
ax.set_xlabel('$u_y(x)$')
plt.show()

