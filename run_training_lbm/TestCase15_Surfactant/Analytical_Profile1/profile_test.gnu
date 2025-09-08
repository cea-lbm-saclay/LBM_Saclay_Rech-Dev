set xlabel "X"
set ylabel "Compos"
set datafile separator ","
##############################################
W = 8.0
eps=  27.76590723574115
k =  1.110636289429646
beta = 1.0
#max_G = -beta*k*(4*eps/(k*(W**2)) +1)/8
valbulk = 0.0005

cb = valbulk/(1-valbulk)

phi(x) = 0.5*(1+tanh(-2*x/W))
dphidx(x) = 4*phi(x)*(1-phi(x))/W
G(x) = -0.5*eps*(dphidx(x)**2)+0.5*k*phi(x)*(phi(x)-1)
comp(x) = cb/(cb + exp(beta*G(x)))

set xrange [-10:10]
set yrange [0.00004:0.0012]
plot "profile_num.csv" u ($3-128):2 w p pt 7 ps 1 lc rgb "blue", comp(x) w l lw 3 lc rgb "red"
pause -1
set xrange [-10:10]
set yrange [0:1]
plot "profile_num.csv" u ($3-128):1 w p pt 7 ps 1 lc rgb "blue", phi(x) w l lw 3 lc rgb "red"
pause -1


