#!/bin/bash
#MSUB -r WetRect1 # nom du job
#MSUB -q a100
#MSUB -Q normal
#MSUB -n 1 # nombre total de tache MPI
#MSUB -N 1  # nombre de nœuds
#MSUB -m work,store,scratch
#MSUB -c 32 # nombre de cœurs par tache
#MSUB -o WetRect1_%j.o # nom du fichier de sortie
#MSUB -e WetRect1_%j.e # nom du fichier d erreur (ici commun avec la sortie)
#MSUB -T 86400 # temps limite à modifier en secondes

# nettoyage des modules charges en interactif et herites par defaut
module purge
# chargement des modules
module load cmake/3.20.3
module load gnu/11.1.0
module load flavor/gnu/cuda-11.4
module load cuda/11.4
##module load mpi/openmpi/4.0.5
module load mpi/openmpi/4.1.4.4
module load flavor/hdf5/parallel
module load hdf5


# echo des commandes lancees
set -x
# execution du code
ccc_mprun /ccc/store/cont002/den/cartalaa/LBM_saclay_Training_v9aout2024/build_cuda_mpi/src/LBM_saclay ./TestCase_Wetting-solid-Rectangle_val1.ini --kokkos-map-device-id-by=mpi_rank

