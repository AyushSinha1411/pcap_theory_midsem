#include <stdio.h>
#include <stdlib.h>

#define N 4  // Matrix dimension (N x N)

// Kernel for row-wise matrix addition (one thread per row)
__global__ void matrixAddRow(int *a, int *b, int *c, int n) {
    int row = threadIdx.x;  // Thread ID represents the row
    if (row < n) {
        for (int col = 0; col < n; col++) {
            c[row * n + col] = a[row * n + col] + b[row * n + col];
        }
    }
}

// Kernel for column-wise matrix addition (one thread per column)
__global__ void matrixAddCol(int *a, int *b, int *c, int n) {
    int col = threadIdx.x;  // Thread ID represents the column
    if (col < n) {
        for (int row = 0; row < n; row++) {
            c[row * n + col] = a[row * n + col] + b[row * n + col];
        }
    }
}

// Kernel for element-wise matrix addition (one thread per element)
__global__ void matrixAddElement(int *a, int *b, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        c[row * n + col] = a[row * n + col] + b[row * n + col];
    }
}

// Function to print matrix
void printMatrix(int *matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d\t", matrix[i * n + j]);
        }
        printf("\n");
    }
}

int main() {
    int size = N * N * sizeof(int);
    int *h_a, *h_b, *h_c;  // Host matrices
    int *d_a, *d_b, *d_c;  // Device matrices
    
    // Allocate host memory
    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);
    
    // Initialize matrices with sample data
    for (int i = 0; i < N * N; i++) {
        h_a[i] = i + 1;
        h_b[i] = i + 2;
    }
    
    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    printf("\nMatrix A:\n");
    printMatrix(h_a, N);
    printf("\nMatrix B:\n");
    printMatrix(h_b, N);
    
    // 1. Row-wise addition
    printf("\n1. Row-wise Addition (one thread per row):\n");
    matrixAddRow<<<1, N>>>(d_a, d_b, d_c, N);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    printMatrix(h_c, N);
    
    // 2. Column-wise addition
    printf("\n2. Column-wise Addition (one thread per column):\n");
    matrixAddCol<<<1, N>>>(d_a, d_b, d_c, N);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    printMatrix(h_c, N);
    
    // 3. Element-wise addition
    printf("\n3. Element-wise Addition (one thread per element):\n");
    dim3 threadsPerBlock(2, 2);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixAddElement<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    printMatrix(h_c, N);
    
    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
