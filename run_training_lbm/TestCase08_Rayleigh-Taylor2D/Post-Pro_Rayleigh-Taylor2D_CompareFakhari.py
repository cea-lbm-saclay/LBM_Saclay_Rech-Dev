import numpy as np
import matplotlib.pyplot as plt
import csv
import os

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 15})


# Le but de ce fichier est de tracer l'évolution de la position du "champignon"
# en fonction du temps adimentionné

path = "Contours/data_"
Step_file = 10

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

        if raw[PosX]=='-64':
            Hauteur_bord.append(float(raw[PosY]))

        elif raw[PosX]=='0':
            Hauteur_Champi.append(float(raw[PosY]))

    num +=Step_file
    size_data += 1
    path_data = path+str(num)+".csv"


# On trace la hauteur en fonction du temps caractéristique
Tcar = 16000
time = 160/Tcar*Step_file * np.arange(0,size_data)
L = 128
height = (np.array(Hauteur_Champi)-2*L)/(L)
height_edge = (np.array(Hauteur_bord)-2*L)/(L)

file_fakhari = open('RT2D_Spike_Ref_Fakhari_PRE2017.dat','r')
data_fakhari = csv.reader(file_fakhari)

x = []
y = []

for row in data_fakhari:
    x.append(float(row[0]))
    y.append(float(row[1]))

file_fakhari_bubble = open('RT2D_Bubble_Ref_Fakhari_PRE2017.dat','r')
data_fakhari_bubble = csv.reader(file_fakhari_bubble)

w1 = []
w2 = []

for row in data_fakhari_bubble:
    w1.append(float(row[0]))
    w2.append(float(row[1]))


plt.scatter(x, y, color = 'red', marker='*', label='Fakhari et al')
plt.scatter(w1, w2, color = 'black', marker='*', label='Fakhari et al')
plt.plot(time,height,"b:^",label="LBM\_Saclay spike")
plt.plot(time,height_edge,"g:o",label="LBM\_Saclay bubble")
plt.legend()
plt.xlabel("$t^{\star}$")
plt.ylabel("$y/L$")
###plt.title("LBM-MRT densité normalisé")
plt.show()



