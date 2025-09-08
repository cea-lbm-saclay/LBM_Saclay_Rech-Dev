import numpy as np
import matplotlib.pyplot as plt
import csv

### Profil de vitesse u_x ###

# Importation des données de l'article de Ghia et al.

data_ghia_x = np.array([
    [1.00000, 1.00000, 1.00000, 1.00000],
    [0.9766, 0.84123, 0.75837, 0.65928],
    [0.9688, 0.78871, 0.68439, 0.57492],
    [0.9609, 0.73722, 0.61756, 0.51117],
    [0.9531, 0.68717, 0.55892, 0.46604],
    [0.8516, 0.23151, 0.29093, 0.33304],
    [0.7344, 0.00332, 0.16256, 0.18719],
    [0.6172, -0.13641, 0.02135, 0.05702],
    [0.5000, -0.20581, -0.11477, -0.06080],
    [0.4531, -0.21090, -0.17119, -0.10648],
    [0.2813, -0.15562, -0.32726, -0.27805],
    [0.1719, -0.10150, -0.24299, -0.38289],
    [0.1016, -0.06434, -0.14612, -0.29730],
    [0.0703, -0.04775, -0.10338, -0.22220],
    [0.0625, -0.04192, -0.09266, -0.20196],
    [0.0547, -0.03717, -0.08186, -0.18109],
    [0.0000, 0.00000, 0.00000, 0.00000]
])

#Re_list = ['100','400','1000']
Re_list = ['1000']
#u_max_list = [6.510416667e-2,1.041666667e-1,2.604166667e-1]
u_max_list = [2.604166667e-1]
L=2.56


# Importation des coordonnées 
y_list = np.array([])
with open('profil_lid_driven_cavity_BGK_1000_Ux.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for k,row in enumerate(reader):
        if row[0]!= 'laplaphi':
            y_list = np.append(y_list, float(row[32]))

            
# Adimensionnement 
y_list = y_list / L     
            
    
    
# Importation des données
data_x = np.zeros((len(y_list),len(Re_list)))

for j,Re in enumerate(Re_list):
    with open('profil_lid_driven_cavity_BGK_' + Re + '_Ux.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for k,row in enumerate(reader):
            if row[0]!= 'laplaphi':
                data_x[k-1,j] = float(row[10])/u_max_list[j]
                
# Graphique

fig,ax = plt.subplots()
fig.set_size_inches(15,11, forward = True)

l1, = ax.plot(data_x[:,0],y_list, color='forestgreen')
#l2, = ax.plot(data_x[:,1],y_list, color='tomato')
#l3, = ax.plot(data_x[:,2],y_list, color='royalblue')

#l4, = ax.plot(data_ghia_x[:,1],data_ghia_x[:,0],  linestyle = 'None', marker='^',markerfacecolor='darkolivegreen',markeredgecolor='darkolivegreen')
#l5, = ax.plot(data_ghia_x[:,2],data_ghia_x[:,0],  linestyle = 'None', marker='o',markerfacecolor='firebrick',markeredgecolor='firebrick')
l6, = ax.plot(data_ghia_x[:,3],data_ghia_x[:,0],  linestyle = 'None', marker='D',markerfacecolor='navy',markeredgecolor='navy')

ax.set_xlim([-0.8,1.0])
ax.set_ylim([0.0,1.0])
ax.set_xlabel('$u_x(y)$')
ax.set_ylabel('$y$')
#ax.legend( (l1,l2,l3,l4,l5,l6), ('LBM BGK $Re = 100$','LBM BGK $Re = 400$','LBM BGK $Re = 1000$', 'Ghia et al. (1982) $Re = 100$', 'Ghia et al. (1982) $Re = 400$', 'Ghia et al. (1982) $Re = 1000$'), loc= 'lower right', shadow = True)
ax.set_title("Profil de vitesse horizontale d'un écoulement 'Lid driven cavity flow' le long de la droite $x=1/2$ par LBM BGK")

plt.show()
