CC = g++
CFLAGS = -g -Wall -std=c++11 -O3 -Wall -shared -std=c++11 -fPIC -I/usr/include/python3.8 `python3.8 -m pybind11 --includes` 
SRCS = src/tracking_class.cpp 
PROG = src/lib/find_marker.so

$(PROG):$(SRCS)
	rm src/lib/find_marker.so
	mkdir -p src/lib
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS)

#clean:
#	rm src/lib/find_marker
