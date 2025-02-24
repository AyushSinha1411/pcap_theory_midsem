// #include <stdio.h>
// #include <mpi.h>

// #define ADDITION 0
// #define SUBTRACTION 1
// #define MULTIPLICATION 2
// #define DIVISION 3

// // Function to simulate each operation
// void calculator_operations(int operation, int a, int b, int *result) {
//     switch (operation) {
//         case ADDITION:
//             *result = a + b;
//             break;
//         case SUBTRACTION:
//             *result = a - b;
//             break;
//         case MULTIPLICATION:
//             *result = a * b;
//             break;
//         case DIVISION:
//             if (b != 0)
//                 *result = a / b;
//             else
//                 *result = -1;  // Return -1 for division by zero
//             break;
//         default:
//             *result = 0;
//             break;
//     }
// }

// int main(int argc, char *argv[]) {
//     int rank, size;
//     int a = 20, b = 5;  // Sample numbers for the calculation
//     int result;
    
//     // Initialize MPI
//     MPI_Init(&argc, &argv);
    
//     // Get the rank of the process and the total number of processes
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);
    
//     // Ensure there are at least 5 processes (one for each operation)
//     if (size < 5) {
//         if (rank == 0) {
//             printf("This program requires at least 5 processes.\n");
//         }
//         MPI_Finalize();
//         return 0;
//     }

//     // Master process (rank 0) sends tasks to other processes
//     if (rank == 0) {
//         for (int i = 1; i <= 4; i++) {
//             MPI_Send(&a, 1, MPI_INT, i, 0, MPI_COMM_WORLD);  // Send 'a'
//             MPI_Send(&b, 1, MPI_INT, i, 0, MPI_COMM_WORLD);  // Send 'b'
//             MPI_Send(&i, 1, MPI_INT, i, 0, MPI_COMM_WORLD);  // Send operation type
//         }

//         // Receive results from each worker
//         for (int i = 1; i <= 4; i++) {
//             MPI_Recv(&result, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//             switch(i) {
//                 case ADDITION:
//                     printf("Addition: %d + %d = %d\n", a, b, result);
//                     break;
//                 case SUBTRACTION:
//                     printf("Subtraction: %d - %d = %d\n", a, b, result);
//                     break;
//                 case MULTIPLICATION:
//                     printf("Multiplication: %d * %d = %d\n", a, b, result);
//                     break;
//                 case DIVISION:
//                     if (result != -1) {
//                         printf("Division: %d / %d = %d\n", a, b, result);
//                     } else {
//                         printf("Division: Division by zero error\n");
//                     }
//                     break;
//                 default:
//                     break;
//             }
//         }
//     } else {
//         // Worker processes (rank 1 to 4)
//         int num1, num2, operation;
        
//         // Receive data from the master
//         MPI_Recv(&num1, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         MPI_Recv(&num2, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         MPI_Recv(&operation, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
//         // Perform the calculation
//         calculator_operations(operation, num1, num2, &result);
        
//         // Send the result back to the master
//         MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
//     }

//     // Finalize MPI
//     MPI_Finalize();
//     return 0;
// }
#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    int a = 10, b = 5;
    int result;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        result = a + b; // Addition
        printf("Process %d: %d + %d = %d\n", rank, a, b, result);
    } else if (rank == 1) {
        result = a - b; // Subtraction
        printf("Process %d: %d - %d = %d\n", rank, a, b, result);
    } else if (rank == 2) {
        result = a * b; // Multiplication
        printf("Process %d: %d * %d = %d\n", rank, a, b, result);
    } else if (rank == 3) {
        result = a / b; // Division
        printf("Process %d: %d / %d = %d\n", rank, a, b, result);
    }

    MPI_Finalize();
    return 0;
}