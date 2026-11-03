# Standalone test of the portability layer: libc only, no PostgreSQL needed.
CFLAGS += -O1 -Wall -I../../src

port_test: port_test.c ../../src/pg_portable.c ../../src/pg_portable.h
	$(CC) $(CFLAGS) -o $@ port_test.c ../../src/pg_portable.c -lpthread

check: port_test
	./port_test

clean:
	rm -f port_test
