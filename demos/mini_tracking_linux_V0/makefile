CC = g++
CFLAGS = -g -shared -Wall -std=c++11 -fPIC -I/usr/include/python3.8
SRCS = tracking_class.cpp 
PROG = find_marker.so

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS)

clean:
	rm -f find_marker.so
