gcc -Wall -std=c99 -I/share/apps/GSL/1.16/include -c mmio.c -o mmio.o
gcc -fopenmp -Wall -std=c99 -g -I/share/apps/GSL/1.16/include/ -c AIMNet.c -o main.o
gcc -fopenmp -Wall -std=c99 -g -L/share/apps/GSL/1.16/lib/  AIMNet.o mmio.o -o AIMNet -lgsl -lgslcblas -lm


gcc -Wall -std=c99 -I/usr/local/include -c mmio.c -o mmio.o
gcc -fopenmp -Wall -std=c99 -g -I/usr/local/include/ -c AIMNet.c -o AIMNet.o
gcc -fopenmp -Wall -std=c99 -g -L/usr/local/lib/  AIMNet.o mmio.o -o AIMNet -lgsl -lgslcblas -lm
