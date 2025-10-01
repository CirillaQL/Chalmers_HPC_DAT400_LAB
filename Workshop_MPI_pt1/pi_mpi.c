#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>

#define N 10000

int main(int argc, char ** argv) {
	MPI_Init(&argc, &argv);
	
	//---- Question 1: Discover the number of processes and rank (ID) of each process 
	int size, rank;
	
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//----
	
	if (rank == 0)
		printf("Computing approximation to pi using N=%d\n", N);
	
	int i;
	double pi = 0.0;		//The final result
	double exact_pi = 0.0;		//Exact computation of pi, for comparison
	double partial_pi = 0.0; 	//Partial result calculated by each process
	
	//---- Question 2: Compute the loop boundaries
	int partitions = N / size;  // Iterations per process
	int start = rank * partitions;  // Loop start
	int end = start + partitions;   // Loop end
	
	if (rank == size - 1) {
		end = N;
	}

	printf("Process %d: start = %d, end = %d\n", rank, start, end);

	//----
	for (i = start; i < end; i++)
		partial_pi = partial_pi + 1.0 / (1.0 + pow((((double)i - 0.5) / (double)N), 2.0));
	
	//---- Question 3: Implement communication
	int tag = 0;
	MPI_Status status;
	double partial_pi_to_recv;

	printf("Process %d: partial_pi = %f\n", rank, partial_pi);
			
	if (rank != 0) {
		// Non-root processes send their partial_pi to rank 0
		MPI_Send(&partial_pi, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
	} else {
		// Rank 0 receives and sums all partial results
		pi = partial_pi;  // Start with rank 0's own contribution
		
		for (i = 1; i < size; i++) {
			MPI_Recv(&partial_pi_to_recv, 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &status);
			pi += partial_pi_to_recv;
		}
	}	
	//----

	// Final result at rank 0
	if (rank == 0) {
		pi = pi * 4.0 / (double)N;
		exact_pi = 4.0 * atan(1.0);
		printf("Pi = %f, Exact pi = %f, Error = %f\n", pi, exact_pi, fabs(100.0 * (pi - exact_pi)/exact_pi));
	}

	MPI_Finalize();
	return 0;
}