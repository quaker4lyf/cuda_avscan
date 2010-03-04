CC=gcc
CFLAGS=-c -Wall

all: cuda_avscan

cuda_avscan: cuda_avscan.o

cuda_avscan.o: cuda_avscan.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *.o cuda_avscan
