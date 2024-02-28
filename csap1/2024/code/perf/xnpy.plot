set logscale y

set multiplot layout 2,1

set key top left
plot 'xnpy_avx_align_unroll.log' using 1:3 w lp title 'Performance [GFLOPS/s]'

set yrange [0.03:0.3]

plot 'xnpy_unroll.log' using 1:2 w lp title 'Time [s]'
