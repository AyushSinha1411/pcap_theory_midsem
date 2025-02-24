#include <stdio.h>
#include <mpi.h>
#include <math.h>

// Function to calculate power of x raised to rank
double power(int x, int rank) {
    return pow(x, rank);
}

int main(int argc, char *argv[]) {
    int rank, size, x = 2; // x is the constant, can be changed to any integer value
    double result;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate the power
    result = power(x, rank);

    // Print the result from each process
    printf("Process %d of %d: %d^%d = %lf\n", rank, size, x, rank, result);

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}