CC = g++
CFLAGS = -std=c++11
LIBS = -lIL

all: flouBox flouGaussien flouSobelOperator flouLaplacienDeGausse

flouBox: flouBox.cpp
	$(CC) $(CFLAGS) $(LIBS) -o flouBox flouBox.cpp

flouGaussien: flouGaussien.cpp
	$(CC) $(CFLAGS) $(LIBS) -o flouGaussien flouGaussien.cpp

flouSobelOperator: flouSobelOperator.cpp
	$(CC) $(CFLAGS) $(LIBS) -o flouSobelOperator flouSobelOperator.cpp

flouLaplacienDeGausse: flouLaplacienDeGausse.cpp
	$(CC) $(CFLAGS) $(LIBS) -o flouLaplacienDeGausse flouLaplacienDeGausse.cpp

clean:
	rm -f flouBox flouGaussien flouSobelOperator flouLaplacienDeGausse
