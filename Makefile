CC = g++
CFLAGS = `pkg-config --cflags opencv`
LDFLAGS = `pkg-config --libs opencv`

all: dettrackdemo

dettrackdemo: objdettrack.o dettrackdemo.cpp
	$(CC) $(CFLAGS) -o $@ -g $+ $(LDFLAGS) -std=c++0x

objdettrack.o: objdettrack.cpp
	$(CC) $(CFLAGS) -o $@ -c -g $+ -std=c++0x

clean:
	rm *.o dettrackdemo
