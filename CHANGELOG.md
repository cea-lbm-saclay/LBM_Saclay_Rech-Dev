# Release Notes

## [Unreleased] 

### Added
- Ajout d’un kernel `NSAC_Surfactant`
- Ajout du fichier `InitConditionsTypes.h` et allègement du fichier `LBM_enums.h`
- Ajout du dossier `TestCase15b_Surfactant` contenant des cas tests pour le problem `NSAC_Surfactant`
- Mise à jour du dossier `compilation` pour la compilation en local, sur orcus et sur topaze

### Fixed
- Problème de compilation lié aux RPATH sur Orcus
- Latence due aux échanges MPI dans `bc_cond`

### Changed
- ORCUS : modification des options CMake pour la compilation multi-GPU sur H100, et mise à jour des modules

## [v1.0.0] - 2025-04-18

### Added
- Implémentation initiale du code `LBM_Saclay`