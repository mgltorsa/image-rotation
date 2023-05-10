#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank of the current process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Print "Hello, world!" from each process
    printf("Hello, world! I'm process %d\n with argv %d", rank, atoi(argv[1]));
    

    // Finalize MPI
    MPI_Finalize();
    return 0;
}