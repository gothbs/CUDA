CC = g++
NVCC = nvcc
CFLAGS = -std=c++11
LIBS = -lIL

all: flouBox flouGaussien flouSobelOperator flouLaplacienDeGausse flouBoxGpu flouGaussienGpu flouSobelOperatorGpu flouLaplacienDeGausseGpu flouBoxOpt flouGaussienOpt flouSobelOperatorOpt flouLaplacienDeGausseOpt

flouBox: flouBox.cpp
	$(CC) $(CFLAGS) -o flouBox flouBox.cpp $(LIBS)

flouGaussien: flouGaussien.cpp
	$(CC) $(CFLAGS) -o flouGaussien flouGaussien.cpp $(LIBS)

flouSobelOperator: flouSobelOperator.cpp
	$(CC) $(CFLAGS) -o flouSobelOperator flouSobelOperator.cpp $(LIBS)

flouLaplacienDeGausse: flouLaplacienDeGausse.cpp
	$(CC) $(CFLAGS) -o flouLaplacienDeGausse flouLaplacienDeGausse.cpp $(LIBS)

flouBoxOptGpu: flouBox.cu
	$(NVCC) $(CFLAGS) -o flouBoxGpu flouBox.cu $(LIBS)

flouGaussienGpu: flouGaussien.cu
	$(NVCC) $(CFLAGS) -o flouGaussienGpu flouGaussien.cu $(LIBS)

flouSobelOperatorGpu: flouSobelOperator.cu
	$(NVCC) $(CFLAGS) -o flouSobelOperatorGpu flouSobelOperator.cu $(LIBS)

flouLaplacienDeGausseGpu: flouLaplacienDeGausse.cu
	$(NVCC) $(CFLAGS) -o flouLaplacienDeGausseGpu flouLaplacienDeGausse.cu $(LIBS)

flouBoxOpt: flouBoxOpt.cu
	$(NVCC) $(CFLAGS) -o flouBoxOpt flouBox.cu $(LIBS)

flouGaussienOpt: flouGaussienOpt.cu
	$(NVCC) $(CFLAGS) -o flouGaussienOpt flouGaussien.cu $(LIBS)

flouSobelOperatorOpt: flouSobelOperatorOpt.cu
	$(NVCC) $(CFLAGS) -o flouSobelOperatorOpt flouSobelOperator.cu $(LIBS)

flouLaplacienDeGausseOpt: flouLaplacienDeGausseOpt.cu
	$(NVCC) $(CFLAGS) -o flouLaplacienDeGausseOpt flouLaplacienDeGausse.cu $(LIBS)

clean:
	rm -f flouBox flouGaussien flouSobelOperator flouLaplacienDeGausse  flouBoxGpu flouGaussienGpu flouSobelOperatorGpu flouLaplacienDeGausseGpu flouBoxOpt flouGaussienOpt flouSobelOperatorOpt flouLaplacienDeGausseOpt
