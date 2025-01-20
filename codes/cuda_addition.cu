#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

const int N = 1000000; // Size of the arrays

// CUDA Kernel to add two arrays
__global__ void add_cuda(const int* a, const int* b, int* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int *h_a, *h_b, *h_c; // Host arrays
    int *d_a, *d_b, *d_c; // Device arrays

    size_t bytes = N * sizeof(int);

    // Allocate memory on host
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    // Initialize arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate memory on GPU
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    auto start = std::chrono::high_resolution_clock::now();
    add_cuda<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Compute duration
    std::chrono::duration<double> duration = end - start;
    std::cout << "CUDA Time taken for addition: " << duration.count() << " seconds\n";

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

