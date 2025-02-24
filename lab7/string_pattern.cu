#include <stdio.h>
#include <string.h>

#define MAX_LENGTH 1024

__global__ void generatePattern(char *S, char *RS, int inputLen, int *positions, int outputLen) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < outputLen) {
        // Find which segment this thread belongs to
        int segment = 0;
        int pos = tid;
        
        // Keep subtracting segment lengths until we find our segment
        while (pos >= positions[segment]) {
            pos -= positions[segment];
            segment++;
        }
        
        // Copy character from appropriate position in source string
        RS[tid] = S[pos];
    }
}

int main() {
    // Input string
    char S[] = "ABC";
    int inputLen = strlen(S);
    
    // Calculate positions for each segment
    int positions[MAX_LENGTH];
    int totalLen = 0;
    
    // First calculate output length and position markers
    for (int i = inputLen; i > 0; i--) {
        positions[inputLen - i] = i;
        totalLen += i;
    }
    
    // Host output string
    char RS[MAX_LENGTH];
    
    // Device memory
    char *d_S, *d_RS;
    int *d_positions;
    
    // Allocate device memory
    cudaMalloc((void**)&d_S, inputLen);
    cudaMalloc((void**)&d_RS, totalLen);
    cudaMalloc((void**)&d_positions, inputLen * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_S, S, inputLen, cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, positions, inputLen * sizeof(int), cudaMemcpyHostToDevice);
    
    // Set block and grid dimensions
    int blockSize = 256;
    int numBlocks = (totalLen + blockSize - 1) / blockSize;
    
    // Launch kernel
    generatePattern<<<numBlocks, blockSize>>>(d_S, d_RS, inputLen, d_positions, totalLen);
    
    // Copy result back to host
    cudaMemcpy(RS, d_RS, totalLen, cudaMemcpyDeviceToHost);
    
    // Null terminate the output string
    RS[totalLen] = '\0';
    
    // Print results
    printf("Input string S: %s\n", S);
    printf("Output string RS: %s\n", RS);
    
    // Free device memory
    cudaFree(d_S);
    cudaFree(d_RS);
    cudaFree(d_positions);
    
    return 0;
}
