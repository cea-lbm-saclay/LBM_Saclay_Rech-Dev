# Introduction

`LBM_saclay` is a mini-application written in C++ designed for teaching and research purpose, and dedicated to students or researchers who wants to learn [Lattice Boltzmann Methods](https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods) (LBM) and their software implementation for a High Performance Computing (HPC). LBM is considered as an alternative numerical method for solving partial differential equations, compared to traditional methods such as [Finite Volumes methods](https://en.wikipedia.org/wiki/Finite_volume_method) or [Finite Elements methods](https://en.wikipedia.org/wiki/Finite_element_method) often used, e.g. in simulating respectively compressible and incompressible Navier-Stokes equations for fluid dynamics.
 
```math
f_{i}(\boldsymbol{x}+\boldsymbol{c}_{i}\delta t,\,t+\delta t)=f_{i}(\boldsymbol{x},\,t)-\frac{1}{\tau_{f}}\left[f_{i}(\boldsymbol{x},\,t)-f_{i}^{eq}(\boldsymbol{x},\,t)\right]
```


