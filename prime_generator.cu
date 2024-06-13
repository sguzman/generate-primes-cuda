#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Error handling macro
#define cudaCheckError() {                              \
    cudaError_t e=cudaGetLastError();                     \
    if(e!=cudaSuccess) {                                  \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                               \
    }                                                     \
}

#include <chrono>

void debugLog(const std::string& message) {
    std::cout << "[DEBUG] " << message << std::endl;
}

__global__ void sieveKernel(bool* isPrime, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 2) return; // 0 and 1 are not primes
    if (isPrime[idx]) {
        for (int j = idx * 2; j < n; j += idx) {
            isPrime[j] = false;
        }
    }
}

void validateInput(int argc, char* argv[], int& number) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <positive integer>" << std::endl;
        exit(EXIT_FAILURE);
    }
    try {
        number = std::stoi(argv[1]);
        if (number < 2) {
            throw std::invalid_argument("Number must be greater than 1.");
        }
    } catch (std::exception& e) {
        std::cerr << "Invalid input: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    int n;
    validateInput(argc, argv, n);
    debugLog("Starting prime number generation.");

    bool* d_isPrime;
    bool* h_isPrime = new bool[n];
    std::fill(h_isPrime, h_isPrime + n, true);

    cudaMalloc((void**)&d_isPrime, n * sizeof(bool));
    cudaCheckError();
    cudaMemcpy(d_isPrime, h_isPrime, n * sizeof(bool), cudaMemcpyHostToDevice);
    cudaCheckError();
    
    auto start = std::chrono::high_resolution_clock::now();
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sieveKernel<<<numBlocks, BLOCK_SIZE>>>(d_isPrime, n);
    cudaDeviceSynchronize();
    cudaCheckError();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    debugLog("CUDA kernel execution completed in " + std::to_string(duration.count()) + " seconds.");

    cudaMemcpy(h_isPrime, d_isPrime, n * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaCheckError();
    cudaFree(d_isPrime);
    cudaCheckError();

    debugLog("Writing primes to file.");
    std::ofstream outfile("primes.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file for writing." << std::endl;
        delete[] h_isPrime;
        exit(EXIT_FAILURE);
    }

    for (int i = 2; i < n; ++i) {
        if (h_isPrime[i]) {
            outfile << i << std::endl;
        }
    }
    outfile.close();
    debugLog("Prime number generation completed.");

    delete[] h_isPrime;
    return 0;
}
