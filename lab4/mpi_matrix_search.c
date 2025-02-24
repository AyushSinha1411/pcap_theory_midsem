#include <stdio.h>
#include <mpi.h>

#define SIZE 3

int main(int argc, char** argv) {
    int rank, size;
    int matrix[SIZE][SIZE];
    int element, count = 0, total_count = 0;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Explicit check for exactly 3 processes
    if (size != 3) {
        if (rank == 0) {
            fprintf(stderr, "Error: Exactly 3 processes required. Current: %d\n", size);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Root process reads matrix and search element
    if (rank == 0) {
        printf("Enter the elements of the 3x3 matrix:\n");
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }
        printf("Enter the element to be searched: ");
        scanf("%d", &element);
    }

    // Broadcast matrix and element to all processes
    MPI_Bcast(matrix, SIZE * SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&element, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process counts occurrences in its assigned row
    int row = rank;  // Each process handles one specific row
    for (int j = 0; j < SIZE; j++) {
        if (matrix[row][j] == element) {
            count++;
        }
    }

    // Reduce to get total count
    MPI_Reduce(&count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root process prints result
    if (rank == 0) {
        printf("The element %d occurs %d times in the matrix.\n", element, total_count);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}