SUMMARY

Validation with the Stefan analytical solution of coupling between composition equation and phase-field equation.

AUTHOR
T. Boutin (2021)

GUIDELINES

A) File "Post-Pro_Compare_Analytical-Solution_Phi.py" computes the analytical solutions of phi and c and compare with solutions computed by LBM_Saclay.
The physical values of parameters are indicated inside the ini file and the python script.

B) After running LBM_Saclay, load the vti files in paraview and perform the steps
- cell data to point data
- plot over line "X axis"
- export csv "composition" and "phi" (clic on "write Time steps" & "write Time steps separately") with name "profil_comp"
- contours phi=0.5
- export csv (clic on "write Time steps" & "write Time steps separately") with name "interface"

C) Run the python script for comparison

> python Post-Pro_Compare_Analytical-Solution.py
