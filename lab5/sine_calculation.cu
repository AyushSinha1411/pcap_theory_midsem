#include <stdio.h>
#include <cuda.h>
#include <cmath>

__global__ void calculateSine(float *angles, float *sines, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        sines[idx] = sin(angles[idx]);
    }
}

int main() {
    int n = 1024; // Length of the array
    size_t size = n * sizeof(float);

    // Allocate memory on the host
    float *h_angles = (float *)malloc(size);
    float *h_sines = (float *)malloc(size);

    // Initialize angles in radians
    for (int i = 0; i < n; i++) {
        h_angles[i] = (float)i * (M_PI / 180.0); // Convert degrees to radians
    }

    // Allocate memory on the device
    float *d_angles, *d_sines;
    cudaMalloc(&d_angles, size);
    cudaMalloc(&d_sines, size);

    // Copy angles from host to device
    cudaMemcpy(d_angles, h_angles, size, cudaMemcpyHostToDevice);

    // Launch kernel with 256 threads per block
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    calculateSine<<<blocksPerGrid, threadsPerBlock>>>(d_angles, d_sines, n);
    cudaMemcpy(h_sines, d_sines, size, cudaMemcpyDeviceToHost);

    // Print results for the first 10 sine values
    printf("Sine values:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_sines[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_angles);
    cudaFree(d_sines);

    // Free host memory
    free(h_angles);
    free(h_sines);

    return 0;
}
