
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void kernel(float* da, float* db, float* dc, int mw, int w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread index is within bounds
    if (i < w) {
        int s = i - (mw / 2);  // The starting index for convolution
        float pv = 0;

        // Perform the convolution
        for (int j = 0; j < mw; j++) {
            if (s + j >= 0 && s + j < w) {
                pv += da[s + j] * db[j];
            }
        }
        dc[i] = pv;
    }
}

int main() {
    int n1, n2;

    printf("Length of the vector: ");
    scanf("%d", &n1);

    printf("Enter the length of the mask: ");
    scanf("%d", &n2);

    float *a = (float *)malloc(n1 * sizeof(float));
    float *b = (float *)malloc(n2 * sizeof(float));
    float *c = (float *)malloc(n1 * sizeof(float));

    float *da, *db, *dc;

    cudaMalloc((void **)&da, n1 * sizeof(float));
    cudaMalloc((void **)&db, n2 * sizeof(float));
    cudaMalloc((void **)&dc, n1 * sizeof(float));

    printf("Enter vector one: ");
    for (int i = 0; i < n1; i++)
        scanf("%f", &a[i]);

    printf("Enter vector two (aka mask): ");
    for (int i = 0; i < n2; i++)
        scanf("%f", &b[i]);

    // Copy data from host to device
    cudaMemcpy(da, a, n1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, n2 * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel configuration: Use more threads per block and a reasonable number of blocks
    int blockSize = 256;  // Set a reasonable block size
    int numBlocks = (n1 + blockSize - 1) / blockSize;  // Calculate number of blocks needed

    // Launch the kernel
    kernel<<<numBlocks, blockSize>>>(da, db, dc, n2, n1);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Synchronize device to ensure kernel execution is complete
    cudaDeviceSynchronize();

    // Copy the result back from device to host
    cudaMemcpy(c, dc, n1 * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Convolution result: \n");
    for (int i = 0; i < n1; i++) {
        printf("%f\t", c[i]);
    }
    printf("\n");

    // Free memory
    free(a);
    free(b);
    free(c);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    return 0;
}
