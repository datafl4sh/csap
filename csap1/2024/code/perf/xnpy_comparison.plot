set term postscript enhanced color
set output "xnpy_clang.eps"
set title "clang 14.0.6"

set yrange [0:25]

set xlabel 'Power'
set ylabel 'GFLOPS/s'

set key top left

mlw = 1.5

plot 'xnpy_naive.log' using 1:3 w l title 'naive' lc rgb '#aa00aa' lw mlw, \
     'xnpy_unroll4.log' using 1:3 w l title 'ur4' lc rgb '#74b72e' lw mlw, \
     'xnpy_unroll8.log' using 1:3 w l title 'ur8' lc rgb '#5dbb63' lw mlw, \
     'xnpy_unroll16.log' using 1:3 w l title 'ur16' lc rgb '#3a5311' lw mlw, \
     'xnpy_unroll32.log' using 1:3 w l title 'ur32' lc rgb '#028a0f' lw mlw, \
     'xnpy_avx_unalign.log' using 1:3 w l title 'AVX ua' lc rgb '#ec9706' lw mlw, \
     'xnpy_avx_align.log' using 1:3 w l title 'AVX a' lc rgb '#c95b0c' lw mlw, \
     'xnpy_avx_unalign_unroll.log' using 1:3 w l title 'AVX ua ur' lc rgb '#ed7117' lw mlw, \
     'xnpy_avx_align_unroll.log' using 1:3 w l title 'AVX a ur' lc rgb '#703803' lw mlw
