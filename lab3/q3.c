#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

int is_vowel(char c) {
    c = tolower(c);
    return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u');
}

int main(int argc, char *argv[]) {
    int rank, size;
    char *string = NULL;
    int string_length, padded_length;
    int local_count = 0;
    int total_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        char temp[1000];
        printf("Enter a string: ");
        fflush(stdout);
        scanf("%s", temp);
        string_length = strlen(temp);
        padded_length = ((string_length + size - 1) / size) * size;
        string = (char*)malloc((padded_length + 1) * sizeof(char));
        strcpy(string, temp);
        for(int i = string_length; i < padded_length; i++) {
            string[i] = ' ';
        }
        string[padded_length] = '\0';
    }

    // Broadcast lengths
    MPI_Bcast(&padded_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&string_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for local string portion
    int local_length = padded_length / size;
    char *local_string = (char*)malloc((local_length + 1) * sizeof(char));

    // Scatter string to all processes
    MPI_Scatter(string, local_length, MPI_CHAR, local_string, local_length, MPI_CHAR, 0, MPI_COMM_WORLD);
    local_string[local_length] = '\0';

    // Count non-vowels in local portion
    for(int i = 0; i < local_length; i++) {
        if (local_string[i] != ' ' && !is_vowel(local_string[i])) {
            local_count++;
        }
    }

    // Synchronize before printing
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int i = 0; i < size; i++) {
        if(rank == i) {
            printf("Process %d found %d non-vowels in substring: %s\n", 
                   rank, local_count, local_string);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Sum up all local counts
    MPI_Reduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nTotal non-vowels: %d\n", total_count);
        fflush(stdout);
        free(string);
    }

    free(local_string);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}