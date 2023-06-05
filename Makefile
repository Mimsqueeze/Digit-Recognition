CC= g++
CFLAGS= -I "./lib/Eigen3" -O3 -DNDEBUG -g -Wall

all: train-network1.exe run-tests1.exe train-network2.exe run-tests2.exe

train-network1.exe: train-network1.o network1.o functions.o
	$(CC) train-network1.o network1.o functions.o -o train-network1.exe

train-network1.o: train-network1.cpp network1.h functions.h
	$(CC) $(CFLAGS) "./src/single hidden layer/train-network1.cpp" -c

train-network1.cpp:

network1.h:

network1.o: network1.cpp network1.h functions.h
	$(CC) $(CFLAGS) "./src/single hidden layer/network1.cpp" -c

network1.cpp:

run-tests1.exe: run-tests1.o network1.o functions.o
	$(CC) run-tests1.o network1.o functions.o -o run-tests1.exe

run-tests1.o: run-tests1.cpp functions.h network1.h
	$(CC) $(CFLAGS) "./src/single hidden layer/run-tests1.cpp" -c

run-tests1.cpp:

train-network2.exe: train-network2.o network2.o functions.o
	$(CC) train-network2.o network2.o functions.o -o train-network2.exe

train-network2.o: train-network2.cpp network2.h functions.h
	$(CC) $(CFLAGS) "./src/double hidden layer/train-network2.cpp" -c

train-network2.cpp:

network2.h:

network2.o: network2.cpp network2.h functions.h
	$(CC) $(CFLAGS) "./src/double hidden layer/network2.cpp" -c

network2.cpp:

run-tests2.exe: run-tests2.o network2.o functions.o
	$(CC) run-tests2.o network2.o functions.o -o run-tests2.exe

run-tests2.o: run-tests2.cpp functions.h network2.h
	$(CC) $(CFLAGS) "./src/double hidden layer/run-tests2.cpp" -c

run-tests2.cpp:

functions.o: functions.cpp functions.h
	$(CC) $(CFLAGS) "./src/functions.cpp" -c

functions.h:

functions.cpp:

clean:
	del *.o *.exe

