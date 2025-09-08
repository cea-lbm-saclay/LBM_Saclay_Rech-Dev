set xlabel "X"
set ylabel "Compos"
set datafile separator ","
set datafile fortran
set grid
#########################################################################################################
# Numerical values for analytical solution corresponding to .ini
W     = 8.0
eps   = 1.0
k     = 0.0
beta  = 15.0
cb0   = 1.1e-5
cb1   = 1.0e-5
gamma = log(cb1/cb0)
#########################################################################################################
# Analytical solution
phi(x)    = 0.5*(1+tanh(-2*x/W))
dphidx(x) = 4*phi(x)*(1-phi(x))/W
G(x)      = 0.5*eps*(dphidx(x)**2) + 0.5*k*phi(x)*(1-phi(x)) - gamma*phi(x)/beta
comp(x)   = cb1/(cb1 + exp(-beta*G(x)))
#########################################################################################################
# Plot comp(x) and phi(x)
#
set xrange [-13:13]
set yrange [0.0000099:0.000012]
plot "profile_2_num.csv" u ($3-128):2 title "LBM" w p pt 6 ps 1.3 lc rgb "blue", comp(x) w l title "Analytical" lw 3 lc rgb "red"
pause -1
set xrange [-10:10]
set yrange [0:1]
plot "profile_2_num.csv" u ($3-128):1 title "LBM" w p pt 6 ps 1.3 lc rgb "blue", phi(x) w l title "Analytical" lw 3 lc rgb "red"
pause -1


