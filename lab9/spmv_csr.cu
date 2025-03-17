#include <stdio.h>
#include <stdlib.h>

// Kernel for SpMV using CSR format
__global__ void spmv_csr_kernel(int *row_ptr, int *col_ind, float *values, 
                               float *x, float *y, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows) {
        float dot = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        
        // Perform dot product for this row
        for (int i = row_start; i < row_end; i++) {
            dot += values[i] * x[col_ind[i]];
        }
        
        y[row] = dot;
    }
}

int main() {
    // Example sparse matrix in dense form (4x4):
    // 5 0 0 1
    // 0 8 0 0
    // 0 0 3 0
    // 4 0 0 9
    
    // CSR representation
    int num_rows = 4;
    int num_cols = 4;
    int nnz = 6; // Number of non-zero elements
    
    // CSR format arrays
    int h_row_ptr[5] = {0, 2, 3, 4, 6}; // Size: num_rows + 1
    int h_col_ind[6] = {0, 3, 1, 2, 0, 3}; // Size: nnz
    float h_values[6] = {5.0f, 1.0f, 8.0f, 3.0f, 4.0f, 9.0f}; // Size: nnz
    
    // Input vector and result vector
    float h_x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_y[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Device arrays
    int *d_row_ptr, *d_col_ind;
    float *d_values, *d_x, *d_y;
    
    // Allocate memory on the device
    cudaMalloc((void**)&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_col_ind, nnz * sizeof(int));
    cudaMalloc((void**)&d_values, nnz * sizeof(float));
    cudaMalloc((void**)&d_x, num_cols * sizeof(float));
    cudaMalloc((void**)&d_y, num_rows * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_row_ptr, h_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, h_col_ind, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, num_cols * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    int blockSize = 256;
    int gridSize = (num_rows + blockSize - 1) / blockSize;
    
    // Launch kernel
    spmv_csr_kernel<<<gridSize, blockSize>>>(d_row_ptr, d_col_ind, d_values, d_x, d_y, num_rows);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Copy result back to host
    cudaMemcpy(h_y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    printf("Sparse Matrix (4x4):\n");
    printf("5 0 0 1\n");
    printf("0 8 0 0\n");
    printf("0 0 3 0\n");
    printf("4 0 0 9\n\n");
    
    printf("Vector X: [%.1f, %.1f, %.1f, %.1f]\n\n", h_x[0], h_x[1], h_x[2], h_x[3]);
    
    printf("Result Vector Y = A * X: [%.1f, %.1f, %.1f, %.1f]\n", 
           h_y[0], h_y[1], h_y[2], h_y[3]);
    
    // Calculate on CPU for verification
    float cpu_y[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < num_rows; i++) {
        for (int j = h_row_ptr[i]; j < h_row_ptr[i + 1]; j++) {
            cpu_y[i] += h_values[j] * h_x[h_col_ind[j]];
        }
    }
    
    printf("CPU Result for verification: [%.1f, %.1f, %.1f, %.1f]\n", 
           cpu_y[0], cpu_y[1], cpu_y[2], cpu_y[3]);
    
    // Free device memory
    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    
    return 0;
}
