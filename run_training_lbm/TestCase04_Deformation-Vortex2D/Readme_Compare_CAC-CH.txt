SUMMARY

Comparison of two interface-capturing models: Conservative Allen-Cahn (CAC) and Cahn-Hilliard (CH)
The velocities u_x and u_y are imposed by an analytical expression (vortex). The purpose is to compare the isolines phi=0.5 for several times of simulations for each model.

***************************************************************************************************************

AUTHOR
A. Cartalade

***************************************************************************************************************

RELEVANT PARAMETERS INSIDE .ini FILE

1) For CAC model: counter_term=1.0 and cahn_hilliard=0.0
2) For CH model : counter_term=0.0 and cahn_hilliard=1.0
3) lambda must be 0 for both (no coupling with diffusion eq)

Values of parameters
[params]
W, Mphi
sigma (only for CH)

[init]
U0 (velocity of vortex)


GUIDELINES
A) Run the test case for for .ini files

> LBM_saclay TestCase04_Interface-Vortex_CAC.ini

and next

> LBM_saclay TestCase04_Interface-Vortex_CH.ini

B) Load vti files in paraview and for both outputs
a. Cell data to point data (Ctrl + space & cell) and apply
b. Clic on icon "Isolines" and set phi=0.5
