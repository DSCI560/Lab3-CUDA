#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

//CUDA KERNEL 

__global__ void conv_kernel(
    const uint8_t* image,
    uint8_t* output,
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
                float pixel = image[iy * width + ix];
                float kval  = kernel[(ky + r) * ksize + (kx + r)];
                sum += pixel * kval;
            }
        }
    }

    if (sum < 0) sum = 0;
    if (sum > 255) sum = 255;
    output[y * width + x] = (uint8_t)sum;
}

//PGM I/O 

uint8_t* read_pgm(const char* filename, int* w, int* h) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        perror("fopen");
        exit(1);
    }

    char magic[3];
    fscanf(f, "%2s", magic);
    if (magic[0] != 'P' || magic[1] != '5') {
        fprintf(stderr, "Only binary PGM (P5) supported\n");
        exit(1);
    }

    fscanf(f, "%d %d", w, h);
    int maxval;
    fscanf(f, "%d", &maxval);
    fgetc(f); // consume newline

    uint8_t* data = (uint8_t*)malloc((*w) * (*h));
    fread(data, 1, (*w) * (*h), f);
    fclose(f);
    return data;
}

void write_pgm(const char* filename, uint8_t* data, int w, int h) {
    FILE* f = fopen(filename, "wb");
    fprintf(f, "P5\n%d %d\n255\n", w, h);
    fwrite(data, 1, w * h, f);
    fclose(f);
}

//MAIN 

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s input.pgm output.pgm\n", argv[0]);
        return 1;
    }

    int width, height;
    uint8_t* h_image = read_pgm(argv[1], &width, &height);
    uint8_t* h_output = (uint8_t*)malloc(width * height);

    // 3x3 blur kernel
    const int ksize = 3;
    float h_kernel[9] = {
        1/9.f, 1/9.f, 1/9.f,
        1/9.f, 1/9.f, 1/9.f,
        1/9.f, 1/9.f, 1/9.f
    };

    uint8_t *d_image, *d_output;
    float *d_kernel;

    CHECK_CUDA(cudaMalloc(&d_image, width * height));
    CHECK_CUDA(cudaMalloc(&d_output, width * height));
    CHECK_CUDA(cudaMalloc(&d_kernel, ksize * ksize * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_image, h_image, width * height, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel,
                          ksize * ksize * sizeof(float),
                          cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    conv_kernel<<<grid, block>>>(
        d_image, d_output, d_kernel, width, height, ksize
    );
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output, d_output,
                          width * height,
                          cudaMemcpyDeviceToHost));

    write_pgm(argv[2], h_output, width, height);

    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_kernel);
    free(h_image);
    free(h_output);

    return 0;
}
