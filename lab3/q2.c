#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size, M, N;
    float *array = NULL;
    float *local_array = NULL;
    float *local_avgs = NULL;
    float *all_avgs = NULL;
    float total_avg = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    N = size;

    if (rank == 0) {
        printf("Enter M (elements per process): ");
        scanf("%d", &M);
        array = (float*)malloc(N * M * sizeof(float));
        printf("Enter %d elements:\n", N * M);
        for(int i = 0; i < N * M; i++) {
            scanf("%f", &array[i]);
        }
    }

    // Broadcast M to all processes
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for local arrays
    local_array = (float*)malloc(M * sizeof(float));
    local_avgs = (float*)malloc(sizeof(float));
    if (rank == 0) all_avgs = (float*)malloc(N * sizeof(float));

    // Scatter array to all processes
    MPI_Scatter(array, M, MPI_FLOAT, local_array, M, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Calculate local average
    float sum = 0;
    for(int i = 0; i < M; i++) {
        sum += local_array[i];
    }
    *local_avgs = sum / M;

    // Gather all averages to root
    MPI_Gather(local_avgs, 1, MPI_FLOAT, all_avgs, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nProcess averages:\n");
        sum = 0;
        for(int i = 0; i < N; i++) {
            printf("Process %d average: %.2f\n", i, all_avgs[i]);
            sum += all_avgs[i];
        }
        total_avg = sum / N;
        printf("\nTotal average: %.2f\n", total_avg);

        free(array);
        free(all_avgs);
    }

    free(local_array);
    free(local_avgs);

    MPI_Finalize();
    return 0;
}