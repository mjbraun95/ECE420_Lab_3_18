# Compile solver
# gcc -Wall -g solver.c Lab3IO.c -o solver -lm -fopenmp
# gcc -Wall -g datagen.c Lab3IO.c -o datagen -lm -fopenmp

CC=gcc
CFLAGS=-Wall -g
LDFLAGS=-lm -fopenmp

# Compile and link the program
all: datagen solver

datagen: datagen.c Lab3IO.c
	$(CC) $(CFLAGS) datagen.c Lab3IO.c -o datagen $(LDFLAGS)

solver: solver.c Lab3IO.c
	$(CC) $(CFLAGS) solver.c Lab3IO.c -o solver $(LDFLAGS)

clean:
	rm -f datagen solver *.o