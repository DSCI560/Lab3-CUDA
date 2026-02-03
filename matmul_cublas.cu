#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

const char* cublasGetStatusString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        default: return "Unknown cuBLAS status";
    }
}

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t s = (call); \
        if (s != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %s\n", __FILE__, __LINE__, cublasGetStatusString(s)); \
            exit(1); \
        } \
    } while (0)

int main(int argc, char* argv[]) {
    int N = 1024;
    if (argc >= 2) {
        N = atoi(argv[1]);
        if (N <= 0) N = 1024;
    }

    // Check CUDA devices
    int devCount = 0;
    cudaError_t devErr = cudaGetDeviceCount(&devCount);
    if (devErr != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(devErr));
        fprintf(stderr, "No CUDA-capable device detected or CUDA runtime not available.\n");
        return 2;
    }
    if (devCount == 0) {
        fprintf(stderr, "No CUDA-capable device detected (cudaGetDeviceCount returned 0).\n");
        return 2;
    }

    // Print device info (use device 0)
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Using device 0: %s (compute capability %d.%d, totalGlobalMem=%zu MB)\n",
           prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024 * 1024));

    size_t bytes = (size_t)N * N * sizeof(float);
    float *A = (float*)malloc(bytes);
    float *B = (float*)malloc(bytes);
    float *C = (float*)malloc(bytes);
    if (!A || !B || !C) {
        fprintf(stderr, "Host malloc failed\n");
        return 3;
    }

    for (size_t i = 0; i < (size_t)N * N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&dA, bytes));
    CHECK_CUDA(cudaMalloc((void**)&dB, bytes));
    CHECK_CUDA(cudaMalloc((void**)&dC, bytes));

    CHECK_CUDA(cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Time the cuBLAS call
    auto t0 = std::chrono::high_resolution_clock::now();

    // For row-major matrices A,B,C (stored in C order), compute C = A * B
    // Use the trick: C^T = B^T * A^T, so pass transposes and swap A/B
    CHECK_CUBLAS(
        cublasSgemm(
            handle,
            CUBLAS_OP_T,  // trans B
            CUBLAS_OP_T,  // trans A
            N, N, N,
            &alpha,
            dB, N,       // B (transposed)
            dA, N,       // A (transposed)
            &beta,
            dC, N
        )
    );

    CHECK_CUDA(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    CHECK_CUDA(cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost));

    printf("N = %d, cuBLAS GPU time = %.3f ms\n", N, elapsed_ms);

    // Basic correctness check (every element should be N since A,B are ones)
    printf("C[0] = %f (expected %f)\n", C[0], (float)N);

    // cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(A); free(B); free(C);

    return 0;
}
