/*
This software package includes source codes in C/C++ that implements the AIMNet algoirthm for joint inference of multiple gene regulatory networks under different conditions. It also includes simulated data and real gene expression data that were used in the paper. 
The program uses OpenMP for parallel implementation and the GNU Scientific Library(GSL) for math (GSL >=1.15).
*/

Compile code in local (two conditions/tissues):
----------
	gcc -Wall -std=c99 -I/GSL_path/include -c mmio.c -o mmio.o
	gcc -fopenmp -Wall -std=c99 -g -I/GSL_path/include -c AIMNet.c -o AIMNet.o
	gcc -fopenmp -Wall -std=c99 -g -L/GSL_path/lib  AIMNet.o mmio.o -o AIMNet -lgsl -lgslcblas -lm

Run program:
----------
	// export the number of threads used in parallel
	// demo_data: the example data, num_condits is the number of tissues/conditions
	// num_condits can be 2,3,4 and 5.
 
	export OMP_NUM_THREADS=5
	./AIMNet demo_simu_data num_condit

Example: 
----------
	For example (set num_condit=2 to run gene regulatory networks infernce under two conditions)
	// GSL-1.16 is installed in the path: /share/apps/GSL/1.16

	gcc -Wall -std=c99 -I/share/apps/GSL/1.16/include -c mmio.c -o mmio.o 
	gcc -fopenmp -Wall -std=c99 -g -I/share/apps/GSL/1.16/include/ -c AIMNet.c -o AIMNet.o 
	gcc -fopenmp -Wall -std=c99 -g -L/share/apps/GSL/1.16/lib/  AIMNet.o mmio.o -o AIMNet -lgsl -lgslcblas -lm 
	export OMP_NUM_THREADS=5 
	./AIMNet demo_simuData 2 


Compile code in local (real data):
----------
	gcc -Wall -std=c99 -I/share/apps/GSL/1.16/include -c mmio.c -o mmio.o
	gcc -fopenmp -Wall -std=c99 -g -I/share/apps/GSL/1.16/include/ -c AIMNet.c -o AIMNet.o
	gcc -fopenmp -Wall -std=c99 -g -L/share/apps/GSL/1.16/lib/  AIMNet.o mmio.o -o AIMNet -lgsl -lgslcblas -lm
	export OMP_NUM_THREADS=5
	./AIMNet demo_real_data 2








