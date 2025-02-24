#include <stdio.h>
#include <string.h>
#include <ctype.h>

// Maximum length constants
#define MAX_STR_LENGTH 1024
#define MAX_WORD_LENGTH 50

// CUDA kernel to count word occurrences
__global__ void countWordOccurrences(char* sentence, int sentenceLen, char* searchWord, int searchWordLen, int* count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < sentenceLen) {
        // Each thread checks if a word starts at its position
        if (tid == 0 || sentence[tid-1] == ' ') {
            // Check if word matches at this position
            bool isMatch = true;
            for (int i = 0; i < searchWordLen && (tid + i) < sentenceLen; i++) {
                if (sentence[tid + i] != searchWord[i]) {
                    isMatch = false;
                    break;
                }
            }
            
            // Verify it's a complete word by checking the next character
            if (isMatch && (tid + searchWordLen == sentenceLen || sentence[tid + searchWordLen] == ' ')) {
                atomicAdd(count, 1);
            }
        }
    }
}

// Helper function to preprocess string (convert to lowercase)
void preprocessString(char* str) {
    for(int i = 0; str[i]; i++) {
        str[i] = tolower(str[i]);
    }
}

int main() {
    char sentence[MAX_STR_LENGTH] = "The quick brown fox jumps over the lazy dog. The fox is quick and brown.";
    char searchWord[MAX_WORD_LENGTH] = "the";
    int count = 0;
    
    // Preprocess strings (convert to lowercase for case-insensitive comparison)
    preprocessString(sentence);
    preprocessString(searchWord);
    
    int sentenceLen = strlen(sentence);
    int searchWordLen = strlen(searchWord);
    
    // Allocate device memory
    char *d_sentence, *d_searchWord;
    int *d_count;
    
    cudaMalloc((void**)&d_sentence, MAX_STR_LENGTH);
    cudaMalloc((void**)&d_searchWord, MAX_WORD_LENGTH);
    cudaMalloc((void**)&d_count, sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_sentence, sentence, MAX_STR_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_searchWord, searchWord, MAX_WORD_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);
    
    // Calculate grid and block dimensions
    int blockSize = 256;
    int numBlocks = (sentenceLen + blockSize - 1) / blockSize;
    
    // Launch kernel
    countWordOccurrences<<<numBlocks, blockSize>>>(d_sentence, sentenceLen, d_searchWord, searchWordLen, d_count);
    
    // Copy result back to host
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print results
    printf("The word '%s' appears %d times in the sentence.\n", searchWord, count);
    
    // Free device memory
    cudaFree(d_sentence);
    cudaFree(d_searchWord);
    cudaFree(d_count);
    
    return 0;
}
