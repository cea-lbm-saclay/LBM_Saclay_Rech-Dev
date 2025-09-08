#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import os

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 15})

# simulation params
dt=5.0*10**-9
step=50
nstep=100

# model params
ms=0.2
ml=0.1
Clinf=0.4
Csinf=0.75
mus=Csinf-ms
mul=Clinf-ml
Ds=0.9
Dl=1.0
mueq=0.4
W=0.0012

def h(phi):
    return phi**2*(3-2*phi)
    
# solution params
xi=0.09242
Bl=(mueq-mul)/special.erfc(xi)
Bs=(mueq-mus)/(special.erfc(xi*(Dl/Ds)**0.5)-2)


# read interface data at every time i*dt*step

interface_position=[]
times=[]

for i in range(nstep+1):
    count=0
    s=0.0
    filename = "interface_"+str(i)+".csv"

    file = open(filename,"r")
    #  next(file)
    # get an average of the interface position (not really needed)
    reader = csv.DictReader(file, delimiter=',')
    for row in reader:
        s+=float(row["Points:0"])
        count+=1
        
    s=s/count
    interface_position.append(s)
    times.append(i*step*dt)

times = np.array(times)
interface_position = np.array(interface_position)
# compute speed through finite difference method
speed = (interface_position[1:-1]-interface_position[0:-2])/(times[1:-1]-times[0:-2])




# read composition data at time dt*step*nstep
compositions=[]
compositionsTh=[]
phi_num=[]
Xs=[]

interface=interface_position[nstep]
time=step*dt*(nstep+1)


count=0
filename = "profil_comp_100.csv"

file = open(filename,"r")

reader = csv.DictReader(file, delimiter=',')
for row in reader:
    comp=float(row["composition"])
    x=float(row["Points:0"])
    phi=float(row["phi"])
    compositions.append(comp)
    Xs.append(x)
    phi_num.append(phi)
    Clt=Clinf+Bl*special.erfc(x/2/(Dl*time)**0.5)
    Cst=Csinf+Bs*(special.erfc(x/2/(Ds*time)**0.5)-2)
    compositionsTh.append(h(phi)*Clt+(1-h(phi))*Cst)


Xs = np.array(Xs)
phi_th=(1+np.tanh(2*(Xs-interface)/W))/2
compositions = np.array(compositions)
compositionsTh = np.array(compositionsTh)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))
fig.suptitle('Comparison with solution of Stefan problem')
fig.subplots_adjust(left=0.04, bottom=None, right=0.98, top=None, wspace=None, hspace=None)

# plot interface position
ax1.plot(times, interface_position, marker='x')
ax1.plot(times, 2*xi*Dl*times**0.5)
ax1.set_title("Interface position over time")
ax1.legend(["LBM_Saclay", "Analytical solution"])
ax1.grid()

# plot interface speed
ax2.plot(times[1:-1], speed, marker='x')
ax2.plot(times[1:-1], Dl*xi/times[1:-1]**0.5)
ax2.legend(["LBM_Saclay", "Analytical solution"])
ax2.set_title("Interface speed over time")
ax2.grid()

# plot composition data
ax3.plot(Xs, compositions,linewidth=3 ,label="Simulation")
ax3.plot(Xs, compositionsTh, linestyle="dashed", color="r", label="Solution")
ax3.plot((interface-W)*np.ones((2,1)), [0, 1],color="k")
ax3.plot((interface+W)*np.ones((2,1)), [0, 1],color="k")
ax3.legend(["LBM_Saclay", "Analytical solution", "Interface limits"])
ax3.grid()
ax3.set_xlim((interface-20*W), (interface+20*W))
ax3.set_ylim(0.35, 0.8)
ax3.set_title("Composition (t="+str(time)+")")

plt.show()
    
    
