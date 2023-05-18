CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS=-lm -lIL

SRCS_CPP = flouBox.cpp flouGaussien.cpp flouSobelOperator.cpp flouLaplacienDeGausse.cpp
SRCS_CU = flouBox.cu flouGaussien.cu flouSobelOperator.cu flouLaplacienDeGausse.cu flouBoxOpt.cu flouGaussienOpt.cu flouSobelOperatorOpt.cu flouLaplacienDeGausseOpt.cu

OBJS_CPP = $(SRCS_CPP:.cpp=.o)
OBJS_CU = $(SRCS_CU:.cu=.o)

all: flou_cpp flou_cu

flou_cpp: $(OBJS_CPP)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

flou_cu: $(OBJS_CU)
	nvcc -o $@ $^ $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o: %.cu
	nvcc -c -o $@ $<

.PHONY: clean

clean:
	rm -f flou_cpp flou_cu $(OBJS_CPP) $(OBJS_CU)
