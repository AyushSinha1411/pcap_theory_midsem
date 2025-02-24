#include <stdio.h>
#include <mpi.h>
#include <string.h>

int main(int argc, char** argv) {
    int rank, size;
    char str[] = "Hello";
    int len = strlen(str);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank < len) {
        if (str[rank] >= 'a' && str[rank] <= 'z') {
            str[rank] -= 32; // Convert to uppercase
        } else if (str[rank] >= 'A' && str[rank] <= 'Z') {
            str[rank] += 32; // Convert to lowercase
        }
        printf("Process %d toggled character: %c\n", rank, str[rank]);
    }

    MPI_Finalize();
    return 0;
}