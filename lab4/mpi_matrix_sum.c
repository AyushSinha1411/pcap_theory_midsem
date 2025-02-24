#include <stdio.h>
#include <mpi.h>

#define SIZE 4

int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[SIZE][SIZE];
    int output[SIZE][SIZE] = {0};

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Only the root process will input the matrix
    if (rank == 0) {
        printf("Enter a 4x4 matrix:\n");
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }
    }

    // Broadcast the matrix to all processes
    MPI_Bcast(matrix, SIZE * SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute the output matrix
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (i == 0) {
                output[i][j] = matrix[i][j]; // First row
            } else {
                for (int k = 0; k <= i; k++) {
                    output[i][j] += matrix[k][j]; // Sum of rows
                }
            }
        }
    }

    // Print the output matrix
    if (rank == 0) {
        printf("Output Matrix:\n");
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                printf("%d ", output[i][j]);
            }
            printf("\n");
        }
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
