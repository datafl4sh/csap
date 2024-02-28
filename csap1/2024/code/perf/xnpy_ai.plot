#set term postscript enhanced color
#set output 'xnpy_ai.eps'

data = 'xnpy_avx_align_unroll.log'

set logscale x
set logscale y

set xrange [0.07:10]

### Machine data
clock = 3.6
membw = 25.6
meas_membw = 20
meas_peak = 21

###
arith_units = 2
simd_width = 4
peak = clock * arith_units * simd_width
peakFMA = peak * 2

set xlabel "Arithmetic intensity"
set ylabel "GFLOPS/s"

set key bottom right
plot data using 5:3 w lp title 'XNPY perf [GFLOPS/s]', \
     data using 5:4 w lp title 'XNPY bw [GB/s]', \
     data using 5:($5)*membw w l title 'Th. peak BW', \
     data using 5:($5)*meas_membw w l title 'Meas. peak BW', \
     peak title 'Th. peak perf', \
     meas_peak title 'Meas. peak perf', \
     peakFMA title 'Th. peak perf (FMA)'

