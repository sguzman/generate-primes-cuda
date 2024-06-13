#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <thread>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define NUM_FILES 4 // Number of files to write to in parallel

// Error handling macro
#define cudaCheckError() {                              \
    cudaError_t e=cudaGetLastError();                     \
    if(e!=cudaSuccess) {                                  \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                               \
    }                                                     \
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

void debugLog(const std::string& message) {
    std::cout << "[DEBUG] " << message << std::endl;
}

void writePrimesToFile(const std::vector<int>& primes, int fileIndex) {
    std::string filename = "primes_" + std::to_string(fileIndex) + ".txt";
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    for (int prime : primes) {
        outfile << prime << std::endl;
    }
    outfile.close();
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
    
    // Using CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sieveKernel<<<numBlocks, BLOCK_SIZE>>>(d_isPrime, n);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    debugLog("CUDA kernel execution completed in " + std::to_string(milliseconds / 1000.0) + " seconds.");

    cudaMemcpy(h_isPrime, d_isPrime, n * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaCheckError();
    cudaFree(d_isPrime);
    cudaCheckError();

    // Split primes into multiple buffers
    std::vector<std::vector<int>> primeBuffers(NUM_FILES);
    for (int i = 2; i < n; ++i) {
        if (h_isPrime[i]) {
            primeBuffers[i % NUM_FILES].push_back(i);
        }
    }

    debugLog("Writing primes to files in parallel.");
    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_FILES; ++i) {
        threads.emplace_back(writePrimesToFile, std::ref(primeBuffers[i]), i);
    }
    for (auto& thread : threads) {
        thread.join();
    }

    debugLog("Prime number generation completed.");

    delete[] h_isPrime;
    return 0;
}

