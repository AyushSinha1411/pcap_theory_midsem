#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA kernel to transform matrix rows
__global__ void transformRows(float* matrix, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        
        // Each row element is raised to the power of (row_index+1)
        // Row 0 elements remain unchanged (power 1)
        // Row 1 elements are squared (power 2)
        // Row 2 elements are cubed (power 3), and so on
        matrix[idx] = powf(matrix[idx], row + 1);
    }
}

int main() {
    int rows, cols;
    
    // Get matrix dimensions from user
    printf("Enter the number of rows: ");
    scanf("%d", &rows);
    printf("Enter the number of columns: ");
    scanf("%d", &cols);
    
    // Allocate memory for matrix on host
    float* h_matrix = (float*)malloc(rows * cols * sizeof(float));
    
    // Read matrix elements from user
    printf("Enter the matrix elements row by row:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            scanf("%f", &h_matrix[i * cols + j]);
        }
    }
    
    // Display the original matrix
    printf("\nOriginal Matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f\t", h_matrix[i * cols + j]);
        }
        printf("\n");
    }
    
    // Allocate memory on the device
    float* d_matrix;
    cudaMalloc((void**)&d_matrix, rows * cols * sizeof(float));
    
    // Copy matrix from host to device
    cudaMemcpy(d_matrix, h_matrix, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions for 2D grid
    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    transformRows<<<gridDim, blockDim>>>(d_matrix, rows, cols);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Copy result back to host
    cudaMemcpy(h_matrix, d_matrix, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Display the transformed matrix
    printf("\nTransformed Matrix:\n");
    for (int i = 0; i < rows; i++) {
        printf("Row %d (power %d):\n", i+1, i+1);
        for (int j = 0; j < cols; j++) {
            printf("%.2f\t", h_matrix[i * cols + j]);
        }
        printf("\n");
    }
    
    // Free memory
    free(h_matrix);
    cudaFree(d_matrix);
    
    return 0;
}
