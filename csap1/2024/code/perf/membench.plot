set logscale x

L1 = 32*1024
L2 = 256*1024
L3 = 8192*1024
MAXBW = 250

set key top left
set xlabel 'Size'
set ylabel 'Bandwidth [GB/s]'

set arrow from L1,0 to L1,MAXBW nohead
set arrow from L2,0 to L2,MAXBW nohead
set arrow from L3,0 to L3,MAXBW nohead
plot 'memcpy.log' using 1:3 w l title 'memcpy()', \
     'memset.log' using 1:3 w l title 'memset()'

