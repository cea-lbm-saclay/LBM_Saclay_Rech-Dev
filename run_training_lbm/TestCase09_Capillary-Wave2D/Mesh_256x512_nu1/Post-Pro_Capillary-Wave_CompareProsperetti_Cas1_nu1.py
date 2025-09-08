import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from math import*
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 15})


# Le but de ce fichier est de tracer l'évolution de la position du "champignon"
# en fonction du temps adimentionné

path = "data_"
Step_file = 1

# Indice dans le data_file
phi     = 0
PosX    = 1
PosY    = 2
PosZ    = 3

# Création du tableau qui va contenir hauteur_champi = g(y)
Hauteur_Champi = []
Hauteur_bord = []

# On localise cette hauteur pour chaque pas de temps par x=0
size_data = 0
num = 0
path_data = path+str(num)+".csv"
while (os.path.isfile(path_data)):
    file = open(path_data,'r')
    data = csv.reader(file)
    for raw in data:

        if raw[PosY]=='0':
            Hauteur_bord.append(float(raw[PosZ]))

        elif raw[PosY]=='128':
            Hauteur_Champi.append(float(raw[PosZ]))

    num +=Step_file
    size_data += 1
    path_data = path+str(num)+".csv"


# On trace la hauteur en fonction du temps caractéristique

L=256
H=512
rhoH=100.0
rhoL=0.12060623666469664
sigma = 4.2e-3
ampl0 = 2.56
k=2*3.14159/L
omega0 = sqrt(sigma*k**3/(rhoH+rhoL))
print("omega0 = ",omega0)
print("size_data =",size_data)

t_star = 10000*omega0* np.arange(0,size_data)
height = (np.array(Hauteur_Champi) - 256)/ampl0

file_physfluids = open('Solution-Prosperetti_Cas1_nu1.dat','r')
data_ref = csv.reader(file_physfluids)

x = []
y = []

for row in data_ref:
    x.append(float(row[0]))
    y.append(float(row[1]))


plt.scatter(x, y, color = 'red', marker='*', label='Ref Prosperetti')
plt.plot(t_star,height,"b:^",label="LBM\_Saclay spike")
plt.legend()
plt.xlabel("$t^{\star}$")
plt.ylabel("$y/a0$")
###plt.title("LBM-MRT densité normalisé")
plt.show()



