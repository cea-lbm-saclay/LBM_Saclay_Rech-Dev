from math import*

print(" ")
print(" ")
print("###############################################################################")
print("###############################################################################")
print("###                COMPARE LES VALEURS DES ANGLES DE CONTACT                ###")
print("###           ISSUES DE LA SIMULATION ET CALCULEES ANALYTIQUEMENT           ###")
print("###             POUR LE CAS TEST DIPHASIQUE AVEC SURFACE SOLIDE             ###")
print("###############################################################################")
print("###############################################################################")

# Paramètres à indiquer en entrée de ce script pour le calcul des angles :
# 1) les valeurs h1, h2 et d sont issues du post-traitement paraview (voir figure dans la présentation)
# 2) Les valeurs de sigma12, sigma1s et sigma2s sont indiquées dans d'entrée de LBM_Saclay
print(" ")
print("La solution analytique est l'Eq. (1.23) page 17 de la réf [1]")
print("[1] de Gennes, Brochard, Quéré, Capillarity and wetting phenomena, Springer (2004)")
print(" ")


###################################################
# Les Cas A1, A2 et A3 correspondent aux valeurs
# mesurées sur un maillage 150x150
# Cas VAL 1 (commenter ou décommenter si nécessaire)
#print("Sigma1s VAL 1")
#h1 = 50
#d = 40
#sigma12 = 0.01
#sigma1s = 0.001
#sigma2s = 0.01

# Cas VAL 2  (commenter ou décommenter si nécessaire)
#print("Sigma1s VAL 2")
#h1 non mesuré
#d  non mesuré
#sigma12 = 0.01
#sigma1s = 0.01
#sigma2s = 0.01

# Cas VAL 3  (commenter ou décommenter si nécessaire)
#print("Sigma1s VAL 3")
#h1 = 50.3515
#d = 14.81778
#sigma12 = 0.01
#sigma1s = 0.01
#sigma2s = 0.008

# Cas VAL 4  (commenter ou décommenter si nécessaire)
#print("Sigma1s VAL 4")
#h1 = 25.9653
#d = 27.73031394
#sigma12 = 0.01
#sigma1s = 0.01
#sigma2s = 0.002

# Cas VAL 5  (commenter ou décommenter si nécessaire)
print("Sigma1s VAL 5")
h1 = 51.357
d  = 34.166
sigma12 = 1.e-4
sigma1s = 5e-5
sigma2s = 1.e-4
###################################################################################################################
print(" ")
print(" ")
print("###############################################################################")
print("PARTIE 1 : ANGLES CALCULÉS PAR LES 3 TENSIONS DE SURFACE")
#
# Valeurs analytiques des angles connaissant les tensions superficielles
# Attention aux conventions des indices :
# Pour la solution analytique Eq. (37) de la réf [1] :
# L'indice 1 est la lentille, la phase 2 est au-dessus et la phase 3 est en-dessous
#
# Pour la simulation numérique : l'indice 0 est la lentille, la phase 1 est au-dessus et la phase 2 est au-dessous
# La solution analytique (37) de la réf [1] est adaptée à la numérotation de la simu LBM_Saclay
#
print ("Les tensions de surface sont :")
print ("sigma12 =",sigma12)
print ("sigma1s =",sigma1s)
print ("sigma2s =",sigma2s)
theta1_analy = acos((sigma2s - sigma1s)/sigma12)
theta1_analy_deg = theta1_analy*180/pi
#Affichage sortie
print(" ")
print("La valeur de theta calculée par les tensions superficielles est :")
print("(voir formules analytiques dans la présentation)")
print("theta1_analy =",theta1_analy_deg," deg")
#print("theta2_analy = ",theta2_analy,"soit ","theta2_analy =",theta2_analy_deg," deg")
###################################################################################################################
print(" ")
print(" ")
print("###############################################################################")
print("PARTIE 2 : ANGLES CALCULÉS GÉOMÉTRIQUEMENT DU POST-PROCESSING PARAVIEW")
print("Les valeurs issues du post-processing paraview sont :")
print("h1 =",h1)
print("d  =",d)

# Les formules correspondent aux Eqs (41) de la réf [1]
# [1] Liang, Shi, Chai, Phys Rev E, 93, 013308 (2016)
#
#theta1_sim = atan(h1/d)
theta1_sim = atan(h1/d)
theta1_sim_deg = theta1_sim*180/pi

#Affichage sortie
print(" ")
print("La valeur de theta est :")
print("(voir formule analytique dans la présentation)")
print("theta1_sim =",theta1_sim_deg," deg")


###################################################################################################################
print(" ")
print(" ")
print("###############################################################################")
print("PARTIE 3 : ERREURS RELATIVES")
print(" ")
print("Les erreurs relatives pour theta1 est :")
err_rel_theta1 = abs(theta1_sim_deg - theta1_analy_deg) / theta1_analy_deg
print("err_rel_theta1 = ",err_rel_theta1)
print("soit : ",err_rel_theta1*100,"%")

print(" ")
print("###############################################################################")
print("###############################################################################")
print("###########################        FIN         ################################")
print("###############################################################################")
print("###############################################################################")
