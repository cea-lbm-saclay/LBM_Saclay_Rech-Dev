#!/bin/bash

# Paramètres du job slurm. Ajuster en particulier le Walltime, le nom du job
# et fichiers de sortie/erreur, et l'adresse mail.
# Plus d'infos à la section 5. du manuel utilisateur d'Orcus.

#SBATCH -N 1                                  # Nombre de nœuds
#SBATCH -n 2                                  # Nombre de taches
#SBATCH --gres=gpu:2                          # nb GPU Orcus (commande à rajouter depuis décembre)
#SBATCH -p gpuq_h100                          # Partitions adaptées au job
#SBATCH -t 12:00:00                           # Walltime
#SBATCH -J cas_test                              # Nom du job
#SBATCH -o cas_test_%j.o                         # Fichier de sortie
#SBATCH -e cas_test_%j.e                         # Fichier d'erreur

# Module nécessaires au fonctionnement de LBM_saclay.
module purge
module load cmake/3.28.3
module load hdf5/nvhpc_25.3_openmpi_5.0.3/1.14.3

# Lance LBM_saclay. Remplacer les deux chemins comme nécessaire.
mpirun -np 2 /home/catA/ac165432/CODEV-TULEAP_LBM_Saclay/LBM_Saclay_Rech-Dev/build_cuda_h100_multigpu/build_NSAC_Comp/src/LBM_saclay $1 
exit 0
