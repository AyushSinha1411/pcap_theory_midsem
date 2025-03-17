#include <stdio.h>
#include <stdlib.h>

// Function to calculate 1's complement of an integer (preserving result in binary form)
__device__ int onesComplement(int num) {
    // Determine the number of bits needed to represent the number
    int numBits = 0;
    int temp = num;
    
    // Count the number of bits
    while (temp > 0) {
        temp >>= 1;
        numBits++;
    }
    
    // Ensure we have at least 1 bit
    if (numBits == 0) {
        numBits = 1;
    }
    
    // Create a mask with all 1's for the number of bits
    int mask = (1 << numBits) - 1;
    
    // Calculate the 1's complement
    return num ^ mask;
}

// Helper function to convert decimal to binary in device code
__device__ int decimalToBinaryDevice(int n) {
    int binaryNum = 0;
    int temp = n;
    int base = 1;
    
    while (temp > 0) {
        int remainder = temp % 2;
        temp = temp / 2;
        binaryNum += remainder * base;
        base = base * 10;
    }
    
    return binaryNum;
}

// CUDA kernel to replace non-border elements with their 1's complement in binary form
__global__ void complementMatrix(int* input, int* output, int* outputBinary, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        
        // Check if the element is a border element
        if (row == 0 || row == rows - 1 || col == 0 || col == cols - 1) {
            // Border element - keep as is
            output[idx] = input[idx];
            outputBinary[idx] = 0; // No binary conversion needed
        } else {
            // Non-border element - replace with 1's complement
            int complement = onesComplement(input[idx]);
            output[idx] = complement;
            outputBinary[idx] = decimalToBinaryDevice(complement);
        }
    }
}

// Helper function to convert decimal to binary
int decimalToBinary(int n) {
    if (n == 0) return 0;
    
    int binaryNum = 0;
    int temp = n;
    int base = 1;
    
    while (temp > 0) {
        int remainder = temp % 2;
        temp = temp / 2;
        binaryNum += remainder * base;
        base = base * 10;
    }
    
    return binaryNum;
}

int main() {
    int rows, cols;
    
    // Get matrix dimensions from user
    printf("Enter the number of rows: ");
    scanf("%d", &rows);
    printf("Enter the number of columns: ");
    scanf("%d", &cols);
    
    // Allocate memory for matrices on host
    int* h_input = (int*)malloc(rows * cols * sizeof(int));
    int* h_output = (int*)malloc(rows * cols * sizeof(int));
    int* h_outputBinary = (int*)malloc(rows * cols * sizeof(int));
    
    // Read matrix elements from user
    printf("Enter the matrix elements row by row:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            scanf("%d", &h_input[i * cols + j]);
        }
    }
    
    // Display the original matrix
    printf("\nInput Matrix A:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d\t", h_input[i * cols + j]);
        }
        printf("\n");
    }
    
    // Allocate memory on the device
    int *d_input, *d_output, *d_outputBinary;
    cudaMalloc((void**)&d_input, rows * cols * sizeof(int));
    cudaMalloc((void**)&d_output, rows * cols * sizeof(int));
    cudaMalloc((void**)&d_outputBinary, rows * cols * sizeof(int));
    
    // Copy input matrix from host to device
    cudaMemcpy(d_input, h_input, rows * cols * sizeof(int), cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions for 2D grid
    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    complementMatrix<<<gridDim, blockDim>>>(d_input, d_output, d_outputBinary, rows, cols);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_outputBinary, d_outputBinary, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Display the output matrix with 1's complement for non-border elements (in binary)
    printf("\nOutput Matrix B (non-border elements replaced with 1's complement in binary):\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
                // Border element - print normally
                printf("%d\t", h_output[i * cols + j]);
            } else {
                // Non-border element - print in bold with binary representation
                printf("\033[1m%d\033[0m\t", h_outputBinary[i * cols + j]);
            }
        }
        printf("\n");
    }
    
    // Explain the 1's complement for better understanding
    printf("\nExplanation:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (i != 0 && i != rows - 1 && j != 0 && j != cols - 1) {
                int dec_complement = h_output[i * cols + j];
                int bin_complement = h_outputBinary[i * cols + j];
                
                printf("%d -> %d (decimal) -> %d (binary form of 1's complement)\n", 
                       h_input[i * cols + j], dec_complement, bin_complement);
            }
        }
    }
    
    // Free memory
    free(h_input);
    free(h_output);
    free(h_outputBinary);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_outputBinary);
    
    return 0;
}
