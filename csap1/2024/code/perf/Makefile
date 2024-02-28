PROGS=axpy xnpy gemv matmult membench cpubench fvol_soa fvol_aos

all: $(PROGS)


SOURCES=axpy.cpp xnpy.cpp gemv.cpp matmult.cpp membench.cpp cpubench.cpp fvol_soa.cpp fvol_aos.cpp utils.cpp
OBJS=$(SOURCES:%.cpp=%.o)
.SUFFIXES: .cpp .o

CXXFLAGS = -g -O3 -std=c++20 -march=native # -Wno-double-promotion -Wno-poison-system-directories -Rpass-analysis=loop-vectorize -Rpass=loop-vectorize
CXXFLAGS += -DHAVE_AVX # Enable if processor has AVX
CXXFLAGS += -DHAVE_FMA # Enable if the processor has FMA
CXXFLAGS += -DHAVE_SILO # Enable if SILO library is available

.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<

axpy: axpy.o utils.o
	$(CXX) -o $@ $^ -lblas

xnpy: xnpy.o utils.o
	$(CXX) -o $@ $^

membench: membench.o utils.o
	$(CXX) -o $@ $^

cpubench: cpubench.o utils.o
	$(CXX) -o $@ $^

gemv: gemv.o utils.o
	$(CXX) -o $@ $^ -lblas

matmult: matmult.o utils.o
	$(CXX) -o $@ $^

fvol_soa: fvol_soa.o utils.o
	$(CXX) -o $@ $^ -lsiloh5

fvol_aos: fvol_aos.o utils.o
	$(CXX) -o $@ $^ -lsiloh5

measurements: xnpy membench cpubench
	./xnpy naive
	./xnpy unroll4
	./xnpy unroll8
	./xnpy unroll16
	./xnpy unroll32
	./xnpy avx_unalign
	./xnpy avx_unalign_unroll
	./xnpy avx_align
	./xnpy avx_align_unroll
	./membench
	./cpubench

.PHONY: clean
clean:
	rm -rf $(OBJS) $(PROGS)
