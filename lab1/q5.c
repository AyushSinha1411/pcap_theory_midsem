#include <stdio.h>
#include <mpi.h>

int factorial(int n) {
    if (n == 0) return 1;
    return n * factorial(n - 1);
}

int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank % 2 == 0) {
        printf("Process %d: Factorial of %d is %d\n", rank, rank, factorial(rank));
    } else {
        printf("Process %d: Fibonacci of %d is %d\n", rank, rank, fibonacci(rank));
    }

    MPI_Finalize();
    return 0;
}