SUMMARY

Test case of Rayleigh-Taylor instability and comparison with the numerical solution of [1]. Validation of NS equations with gravity and surface tension force, coupled to the interface-capturing model (CAC equation).

AUTHORS
Théo DUEZ (2022)
A. CARTALADE

GUIDELINES

A) Both files "RT2D_Bubble_Ref_Fakhari_PRE2017.csv" & "RT2D_Spike_Ref_Fakhari_PRE2017.csv" contain t* and y positions of bubble point (1st file) and spike (2nd file). They have been digitalized from Fig 6 of reference [1].

B) The input parameters inside the .ini file of LBM_Saclay correspond to those calculated in the python script "Pre-Pro_InputParam_Rayleigh-Taylor.py"

> python Pre-Pro_InputParam_Rayleigh-Taylor.py

C) After running LBM_Saclay, load the vti files in paraview and perform the three steps
- cell data to point data
- contours phi=0.5
- Save data: export csv (clic on "write Time steps" & "write Time steps separately") inside folder "Contours" with name "data"


D) Run the python script for comparison:

> python Post-Pro_Rayleigh-Taylor2D_CompareFakhari.py

REFERENCE
[1] Fakhari et al, PHYSICAL REVIEW E 96, 053301 (2017). doi 10.1103/PhysRevE.96.053301
