#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define TILE_WIDTH 16

//TILED MATRIX MULTIPLICATION (UNCHANGED)

__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {

        if (Row < N && (m * TILE_WIDTH + tx) < N)
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;

        if (Col < N && (m * TILE_WIDTH + ty) < N)
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + Col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];

        __syncthreads();
    }

    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}

//Exposed C API for Python (Matrix Multiply) 
extern "C"
void gpu_matrix_multiply(float *h_A, float *h_B, float *h_C, int N) {
    size_t size = N * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH,
              (N + TILE_WIDTH - 1) / TILE_WIDTH);

    matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// IMAGE CONVOLUTION (NEW ADDITION)
  

__global__ void convolutionKernel(
    const uint32_t* image,
    uint32_t* output,
    const float* kernel,
    int width,
    int height,
    int ksize
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int r = ksize / 2;
    float sum = 0.0f;

    for (int ky = -r; ky <= r; ky++) {
        for (int kx = -r; kx <= r; kx++) {
            int ix = x + kx;
            int iy = y + ky;

            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                float pixel = (float)image[iy * width + ix];
                float kval  = kernel[(ky + r) * ksize + (kx + r)];
                sum += pixel * kval;
            }
        }
    }

    sum = fminf(fmaxf(sum, 0.0f), 255.0f);
    output[y * width + x] = (uint32_t)sum;
}

// Exposed C API for Python (Convolution) 
extern "C"
void gpu_convolution(
    const uint32_t* h_image,
    uint32_t* h_output,
    const float* h_kernel,
    int width,
    int height,
    int ksize
) {
    uint32_t *d_image, *d_output;
    float *d_kernel;

    size_t img_bytes = width * height * sizeof(uint32_t);
    size_t ker_bytes = ksize * ksize * sizeof(float);

    cudaMalloc(&d_image, img_bytes);
    cudaMalloc(&d_output, img_bytes);
    cudaMalloc(&d_kernel, ker_bytes);

    cudaMemcpy(d_image, h_image, img_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, ker_bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    convolutionKernel<<<grid, block>>>(
        d_image, d_output, d_kernel,
        width, height, ksize
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, img_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_kernel);
}
