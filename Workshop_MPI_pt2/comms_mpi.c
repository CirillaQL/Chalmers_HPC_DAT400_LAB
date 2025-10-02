#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char ** argv) {
	int size, rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//----Question 1: Create new communicators, for even and odd processes
	int color; 		//TODO: Assign a color (even/odd) to each process, according to its rank

	MPI_Comm new_comm;	//TODO: Create a new communicator based on the color of each process

	//----
	

	//----Question 2: A process with a new rank 0 on each communicator should now broadcast its color 
	int new_rank;			//TODO: Find the rank of each process in the new communicator it belongs to

	int communicator_color; 	//TODO: The color of new rank 0 to be broadcasted
	if (new_rank == 0)
		communicator_color = color;
	//TODO: Broadcast the color to all processes in the new communicator

	//----
	
	//----Question 3: Each process should print its rank in the "world" and in the new communicator and the received color
	
	//----
	MPI_Finalize();
	return 0;
}
