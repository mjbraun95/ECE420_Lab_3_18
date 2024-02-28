CC=gcc
CFLAGS=-Wall -g

all: datagen Lab3IO

datagen: datagen.c
	$(CC) $(CFLAGS) -o datagen datagen.c

Lab3IO: Lab3IO.c
	$(CC) $(CFLAGS) -o Lab3IO Lab3IO.c

clean:
	rm -f datagen Lab3IO
