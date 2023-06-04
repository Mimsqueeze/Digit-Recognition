CC= g++
CFLAGS= -I "./lib/Eigen3" -O3 -DNDEBUG

all: train-network1.exe run-tests1.exe

train-network1.exe: train-network1.o network1.o functions.o
	$(CC) train-network1.o network1.o functions.o -o train-network1.exe

train-network1.o: train-network1.cpp network1.h functions.h
	$(CC) $(CFLAGS) "./src/single hidden layer/train-network1.cpp" -c

train-network1.cpp:

network1.h:

network1.o: network1.cpp network1.h functions.h
	$(CC) $(CFLAGS) "./src/single hidden layer/network1.cpp" -c

network1.cpp:

functions.o: functions.cpp functions.h
	$(CC) $(CFLAGS) "./src/functions.cpp" -c

functions.h:

functions.cpp:

run-tests1.exe: run-tests1.o network1.o functions.o
	$(CC) run-tests1.o network1.o functions.o -o run-tests1.exe

run-tests1.o: run-tests1.cpp functions.h network1.h
	$(CC) $(CFLAGS) "./src/single hidden layer/run-tests1.cpp" -c

run-tests1.cpp:

clean:
	del *.o *.exe

