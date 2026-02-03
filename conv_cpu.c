#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

//CPU CONVOLUTION FUNCTION (UNCHANGED)
   

void cpu_convolution(
    const uint32_t* image,
    uint32_t* output,
    const float* kernel,
    int width,
    int height,
    int ksize
) {
    int r = ksize / 2;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
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

            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;
            output[y * width + x] = (uint32_t)sum;
        }
    }
}

//SIMPLE PGM IMAGE I/O
   

uint32_t* read_pgm(const char* filename, int* width, int* height) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        perror("fopen");
        return NULL;
    }

    char magic[3];
    fscanf(f, "%2s", magic);
    if (strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Only binary PGM (P5) supported\n");
        fclose(f);
        return NULL;
    }

    fscanf(f, "%d %d", width, height);
    int maxval;
    fscanf(f, "%d", &maxval);
    fgetc(f);  // consume newline

    uint8_t* temp = (uint8_t*)malloc((*width) * (*height));
    fread(temp, 1, (*width) * (*height), f);
    fclose(f);

    uint32_t* image = (uint32_t*)malloc((*width) * (*height) * sizeof(uint32_t));
    for (int i = 0; i < (*width) * (*height); i++)
        image[i] = temp[i];

    free(temp);
    return image;
}

void write_pgm(const char* filename, const uint32_t* image, int width, int height) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        perror("fopen");
        return;
    }

    fprintf(f, "P5\n%d %d\n255\n", width, height);

    for (int i = 0; i < width * height; i++) {
        uint8_t val = (uint8_t)image[i];
        fwrite(&val, 1, 1, f);
    }

    fclose(f);
}

//MAIN DRIVER
   

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.pgm output.pgm\n", argv[0]);
        return 1;
    }

    int width, height;
    uint32_t* image = read_pgm(argv[1], &width, &height);
    if (!image) return 1;

    uint32_t* output = (uint32_t*)malloc(width * height * sizeof(uint32_t));

    //Example filter: Sobel X 
    float kernel[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };

    cpu_convolution(image, output, kernel, width, height, 3);

    write_pgm(argv[2], output, width, height);

    free(image);
    free(output);

    return 0;
}
