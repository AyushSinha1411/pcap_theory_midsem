#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;
    char input_word[100];
    char output_word[100] = {0};
    int word_length;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process reads the input word
    if (rank == 0) {
        printf("Enter a word: ");
        scanf("%s", input_word);
        word_length = strlen(input_word);

        // Validate that number of processes matches word length
        if (word_length != size) {
            printf("Error: Number of processes must match word length\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast the input word and its length to all processes
    MPI_Bcast(input_word, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&word_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process handles its corresponding character
    char current_char = input_word[rank];
    int repeat_count = rank + 1;  // Repeat count increases with rank

    // Create local output for each process
    char local_output[10] = {0};
    for (int i = 0; i < repeat_count; i++) {
        local_output[i] = current_char;
    }

    // Gather local outputs to root process
    char gathered_output[100][10];
    MPI_Gather(local_output, 10, MPI_CHAR, 
               gathered_output, 10, MPI_CHAR, 
               0, MPI_COMM_WORLD);

    // Root process assembles the final output
    if (rank == 0) {
        // Reset output word
        memset(output_word, 0, sizeof(output_word));

        // Combine gathered outputs
        for (int i = 0; i < word_length; i++) {
            strcat(output_word, gathered_output[i]);
        }

        // Display the result
        printf("Input word: %s\n", input_word);
        printf("Output word: %s\n", output_word);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}