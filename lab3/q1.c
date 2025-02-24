#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int main(int argc, char *argv[]) {
    int rank, size, N;
    int *values = NULL;
    int *results = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter N (number of values): ");
        scanf("%d", &N);
        values = (int*)malloc(N * sizeof(int));
        results = (int*)malloc(N * sizeof(int));
        
        printf("Enter %d values:\n", N);
        for(int i = 0; i < N; i++) {
            scanf("%d", &values[i]);
        }
    }

    // Broadcast N to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter values to all processes
    int local_N = N/size;
    int *local_values = (int*)malloc(local_N * sizeof(int));
    int *local_results = (int*)malloc(local_N * sizeof(int));

    MPI_Scatter(values, local_N, MPI_INT, local_values, local_N, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate factorial for local values
    for(int i = 0; i < local_N; i++) {
        local_results[i] = factorial(local_values[i]);
    }

    // Gather results back to root
    MPI_Gather(local_results, local_N, MPI_INT, results, local_N, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nFactorials:\n");
        for(int i = 0; i < N; i++) {
            printf("%d! = %d\n", values[i], results[i]);
        }
        free(values);
        free(results);
    }

    free(local_values);
    free(local_results);

    MPI_Finalize();
    return 0;
}