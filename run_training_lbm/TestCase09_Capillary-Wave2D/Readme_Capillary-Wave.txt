SUMARY

Test case of capillary wave and comparison with the analytical solution of Prosperetti. Validation of NS equations with surface tension force and without gravity, coupled to interface-capturing model (CAC equation).

AUTHORS
Hoel KERAUDREN (2023)
A. CARTALADE

GUIDELINES

A) File "Solution-Prosperetti.dat" contains t* and y computed by the analytical solution of Prosperetti. That solution was obtained for the following input parameters set inside each python script

B) After running LBM_Saclay, load the vti files in paraview and perform the three steps
- cell data to point data
- contours phi=0.5
- export csv (clic on "write Time steps" & "write Time steps separately") inside folder "Contours" with name "data"

C) Run the python script for comparison

> python Post-Pro_Capillary-Wave_CompareProsperetti.py
