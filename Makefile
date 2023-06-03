CC= g++
CFLAGS= -I "./lib/Eigen3" -O3 -DNDEBUG

all: network.exe run-tests.exe

network.exe: network.o functions.o
	$(CC) network.o functions.o -o network.exe

network.o: network.cpp functions.h
	$(CC) $(CFLAGS) ./src/network.cpp -c

network.cpp:

functions.o: functions.cpp functions.h
	$(CC) $(CFLAGS) ./src/functions.cpp -c

functions.h:

functions.cpp:

run-tests.exe: run-tests.o functions.o
	$(CC) run-tests.o functions.o -o run-tests.exe

run-tests.o: run-tests.cpp functions.h
	$(CC) $(CFLAGS) ./src/run-tests.cpp -c

run-tests.cpp:

clean:
	del *.o *.exe

