build:
	mkdir build

build/cinterp.o: build c_src/cinterp.c
	gcc -c -fPIC -o build/cinterp.o -O3 c_src/cinterp.c

build/library.so: build/cinterp.o
	gcc -o build/library.so build/cinterp.o -shared
