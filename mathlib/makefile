build/welzl.o: c_src/welzl.c
	gcc -c -fPIC -o build/welzl.o -O3 c_src/welzl.c

build/library.so: build/welzl.o
	gcc -o build/library.so build/welzl.o -L/usr/lib/libblas/ -lblas -L/usr/lib/ -llapack -shared
