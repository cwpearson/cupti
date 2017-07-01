TARGETS = prof.so

all: $(TARGETS)

OBJECTS = prof.o callbacks.o value.o values.o allocation.o allocations.o extent.o set_device.o

CXX=g++
CXXFLAGS= -std=c++11 -g -Wall -Wextra -Wshadow -Wpedantic -fPIC
INC = -I/usr/local/cuda/include -I/usr/local/cuda/extras/CUPTI/include
LIB = -L/usr/local/cuda/extras/CUPTI/lib64 -lcupti -ldl

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(INC) $^ -c -o $@	

prof.so: $(OBJECTS)
	$(CXX) -shared $^ $(LIB) -o $@

clean:
	rm -f *.o cprof prof.so
