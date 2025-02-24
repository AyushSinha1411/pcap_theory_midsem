#include <stdio.h>
#include <mpi.h>

// Function to calculate factorial
long long factorial(int n) {
    if (n == 0 || n == 1) return 1;
    long long result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

int main(int argc, char** argv) {
    int rank, size, N;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Error handling for number of processes
    if (size < 1) {
        if (rank == 0) {
            fprintf(stderr, "Error: At least one process is required.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Set N (number of factorials to calculate)
    if (rank == 0) {
        printf("Enter the value of N: ");
        scanf("%d", &N);
    }

    // Broadcast N to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate factorial for each rank
    long long local_factorial = factorial(rank + 1);

    // Use MPI_Scan to calculate the running total of factorials
    long long total_factorial;
    MPI_Scan(&local_factorial, &total_factorial, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    // Print the results
    printf("Process %d: Factorial of %d is %lld, Running total is %lld\n", rank, rank + 1, local_factorial, total_factorial);

    MPI_Finalize();
    return 0;
}
