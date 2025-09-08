#!/bin/bash

# Paramètres du job slurm. Ajuster en particulier le Walltime, le nom du job
# et fichiers de sortie/erreur, et l'adresse mail.
# Plus d'infos à la section 5. du manuel utilisateur d'Orcus.

#SBATCH -n 1                                  # Nombre de taches
#SBATCH --gres=gpu:1
#SBATCH -p gpuq_h100                          # Partitions adaptées au job
##SBATCH --qos=normal                          # QOS
#SBATCH -t 12:00:00                           # Walltime
#SBATCH -J myjob                              # Nom du job
#SBATCH -o myjob_%j.o                         # Fichier de sortie
#SBATCH -e myjob_%j.e                         # Fichier d'erreur
#SBATCH --mail-user=user@cea.fr               # Adresse email
#SBATCH --mail-type=begin,end,fail            # Envoi d'email à l'exécution, fin et échec du job

# Module nécessaires au fonctionnement de LBM_saclay.
#module load gcc/13.3.0 cuda/12.4.0 hdf5/gcc_13.3.0_openmpi_4.1.6/1.14.3
module load cmake/3.28.3
module load hdf5/nvhpc_25.3_openmpi_5.0.3/1.14.3
# Lance LBM_saclay. Remplacer les deux chemins comme nécessaire.
srun /home/catA/ac165432/CODEV-TULEAP_LBM_Saclay/LBM_Saclay_Rech-Dev/build_cuda_h100/build_NSAC_Surfactant/src/LBM_saclay $1
exit 0
