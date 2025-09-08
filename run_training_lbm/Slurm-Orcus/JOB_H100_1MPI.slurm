#!/bin/bash

# Paramètres du job slurm. Ajuster en particulier le Walltime, le nom du job
# et fichiers de sortie/erreur, et l'adresse mail.
# Plus d'infos à la section 5. du manuel utilisateur d'Orcus.

#SBATCH -n 1                                  # Nombre de taches
#SBATCH --gres=gpu:1			      # Nb de GPUs reserves
#SBATCH -p gpuq_h100                          # Partitions adaptées au job
#SBATCH -t 24:00:00                           # Walltime
#SBATCH -J Job                                # Nom du job
#SBATCH -o Job%j.o                            # Fichier de sortie
#SBATCH -e Job%j.e                            # Fichier d'erreur
#SBATCH --mail-user=user@cea.fr               # Adresse email
#SBATCH --mail-type=begin,end,fail            # Envoi d'email à l'exécution, fin et échec du job

# Module nécessaires au fonctionnement de LBM_saclay.
module load cmake/3.28.3 hdf5/nvhpc_25.3_openmpi_5.0.3/1.14.3
# Lance LBM_saclay. Remplacer les deux chemins comme nécessaire.
mpirun -np 1 /home/catA/ac165432/CODEV-TULEAP_LBM_Saclay/LBM_Saclay_Rech-Dev/build_cuda_h100_multigpu/build_NSAC_Comp/src/LBM_saclay $1 --kokkos-map-device-id-by=mpi_rank
exit 0
