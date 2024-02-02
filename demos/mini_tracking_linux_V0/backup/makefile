CC = g++
CFLAGS = -g -Wall -shared -std=c++11 -fPIC -I/usr/include/python3.8 `python3.8 -m pybind11 --includes` 
SRCS = tracking_class.cpp 
PROG = find_marker.so

$(PROG):$(SRCS)
	mkdir -p src/lib
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS)

clean:
	rm -f src/lib/find_marker.so
