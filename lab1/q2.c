#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Check if the rank is even or odd
    if (rank % 2 == 0) {
        printf("Rank %d: Hello\n", rank); // Even ranks print "Hello"
    } else {
        printf("Rank %d: World\n", rank); // Odd ranks print "World"
    }

    // Finalize MPI environment
    MPI_Finalize();
    
    return 0;
}
