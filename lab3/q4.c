#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, size;
    char *s1 = NULL, *s2 = NULL, *result = NULL;
    int length, padded_length;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        char temp1[1000], temp2[1000];
        printf("Enter first string: ");
        fflush(stdout);
        scanf("%s", temp1);
        printf("Enter second string: ");
        fflush(stdout);
        scanf("%s", temp2);

        if (strlen(temp1) != strlen(temp2)) {
            printf("Strings must be of equal length\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        length = strlen(temp1);
        padded_length = ((length + size - 1) / size) * size;

        s1 = (char*)malloc((padded_length + 1) * sizeof(char));
        s2 = (char*)malloc((padded_length + 1) * sizeof(char));
        result = (char*)malloc((padded_length * 2 + 1) * sizeof(char));
        
        strcpy(s1, temp1);
        strcpy(s2, temp2);
        
        for(int i = length; i < padded_length; i++) {
            s1[i] = ' ';
            s2[i] = ' ';
        }
        s1[padded_length] = '\0';
        s2[padded_length] = '\0';
    }

    // Broadcast lengths
    MPI_Bcast(&padded_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_length = padded_length / size;
    char *local_s1 = (char*)malloc((local_length + 1) * sizeof(char));
    char *local_s2 = (char*)malloc((local_length + 1) * sizeof(char));
    char *local_result = (char*)malloc((local_length * 2 + 1) * sizeof(char));

    // Scatter both strings
    MPI_Scatter(s1, local_length, MPI_CHAR, local_s1, local_length, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(s2, local_length, MPI_CHAR, local_s2, local_length, MPI_CHAR, 0, MPI_COMM_WORLD);

    local_s1[local_length] = '\0';
    local_s2[local_length] = '\0';

    // Merge local strings
    int k = 0;
    for(int i = 0; i < local_length && (rank * local_length + i) < length; i++) {
        local_result[k++] = local_s1[i];
        local_result[k++] = local_s2[i];
    }
    while(k < local_length * 2) {
        local_result[k++] = ' ';
    }

    // Gather results
    MPI_Gather(local_result, local_length * 2, MPI_CHAR, result, local_length * 2, MPI_CHAR, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        result[length * 2] = '\0';
        printf("\nResultant string: %s\n", result);
        fflush(stdout);
        free(s1);
        free(s2);
        free(result);
    }

    free(local_s1);
    free(local_s2);
    free(local_result);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}