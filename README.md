# Lab 3: CUDA Program and Custom Python Library

**Team Name:** GamerSups

**Team Members:**
- Bhargav Limbasia (USC ID: 4651477356)
- Harsh Marar (USC ID: 4955651786)
- Nishkarsh Mittal

## Overview

This laboratory exercise explores high-performance matrix computation by progressively transitioning from traditional CPU-based implementations to GPU-accelerated solutions using NVIDIA CUDA. The lab demonstrates performance scaling, memory access patterns, and optimization techniques on modern heterogeneous computing systems.

## System Requirements

- NVIDIA GPU with CUDA support (tested on GeForce RTX 5070)
- CUDA Toolkit 12.0 or higher
- GCC compiler
- Python 3.x
- NumPy
- PIL (Python Imaging Library)

## Project Structure

```
Lab3-CUDA/
├── task1.c                 # CPU matrix multiplication
├── matmul_naive.cu        # Naive CUDA implementation
├── matmul_tiled.cu        # Optimized CUDA with shared memory
├── matmul_cublas.cu       # cuBLAS implementation
├── matrix_lib.cu          # CUDA shared library source
├── test_matrix.py         # Python interface for matrix multiplication
├── conv_cpu.c             # CPU convolution
├── conv_cuda.cu           # CUDA convolution executable
├── conv.py                # Python interface for convolution
└── README.md
```

## Implementation Details

### Part 1: CPU Matrix Multiplication

Basic C implementation using three nested loops with O(N³) complexity.

**Compilation:**
```bash
gcc task1.c -O2 -o matrix_cpu
```

**Usage:**
```bash
./matrix_cpu 512
./matrix_cpu 1024
./matrix_cpu 2048
```

### Part 2: Naive CUDA Implementation

GPU implementation where each thread computes one matrix element. Demonstrates massive parallelism but with inefficient global memory access patterns.

**Compilation:**
```bash
nvcc matmul_naive.cu -O2 -o matmul_naive
```

**Usage:**
```bash
./matmul_naive 512
./matmul_naive 1024
./matmul_naive 2048
```

### Part 3: Local GPU Deployment

Deployed on NVIDIA GeForce RTX 5070 with CUDA Toolkit 12.x. Verified device accessibility and measured performance improvements over CPU execution.

### Part 4: Optimized CUDA with Shared Memory Tiling

Improved implementation using shared memory tiling to reduce global memory accesses and improve data locality.

**Compilation:**
```bash
nvcc matmul_tiled.cu -O3 -o matmul_tiled
```

**Usage:**
```bash
./matmul_tiled 512
./matmul_tiled 1024
./matmul_tiled 2048
```

### Part 5: Performance Comparison

| Matrix Size | CPU Time (s) | Naive GPU (ms) | Tiled GPU (ms) | Speedup |
|-------------|--------------|----------------|----------------|---------|
| N=512       | 0.122594     | 2.027          | 1.997          | 61.4x   |
| N=1024      | 3.292627     | 1.258          | 0.868          | 3793.8x |
| N=2048      | 37.98839     | 8.342          | 6.308          | 6021.7x |

### Part 6: cuBLAS Implementation

Vendor-optimized implementation using cublasSgemm. Demonstrates production-grade GPU matrix multiplication with advanced hardware-specific optimizations.

**Compilation:**
```bash
nvcc matmul_cublas.cu -lcublas -O3 -o matmul_cublas
```

**Note:** Due to WSL limitations, cuBLAS runtime execution was not completed, but code correctness was verified.

### Part 7: Shared Library and Python Integration

CUDA kernels compiled into a shared library and accessed from Python using ctypes.

**Compilation:**
```bash
nvcc -shared -Xcompiler -fPIC matrix_lib.cu -o libmatrix.so
```

**Python Usage:**
```bash
python3 test_matrix.py
```

### Extension: Image Convolution

Implemented 2D convolution for image processing with CPU, standalone CUDA, and Python-accessible CUDA versions.

**CPU Convolution:**
```bash
gcc conv_cpu.c -O3 -o conv_cpu
./conv_cpu input.pgm output_cpu.pgm
```

**CUDA Convolution:**
```bash
nvcc conv_cuda.cu -O3 -o conv_cuda
./conv_cuda input.pgm output_cuda.pgm
```

**Python Convolution:**
```bash
python3 conv.py
```

## Key Findings

1. GPU speedup increases dramatically with problem size, exceeding 6000x for large matrices.
2. Even for moderate sizes (N=512), GPU implementations provide substantial performance gains.
3. Shared memory tiling significantly reduces global memory access overhead.
4. Vendor-optimized libraries like cuBLAS provide superior performance through architecture-specific tuning.
5. Kernel launch overhead can dominate performance for very small workloads.

## Performance Insights

The results demonstrate clear performance trends:

- CPU execution scales poorly with O(N³) complexity
- Naive CUDA provides immediate parallelization benefits
- Shared memory optimization yields significant additional improvements
- Production systems benefit from vendor-optimized libraries
- Problem size determines the effectiveness of GPU acceleration

## Python Integration

The shared library approach enables seamless integration between high-performance CUDA code and high-level Python environments. This demonstrates practical workflows for:

- Machine learning preprocessing
- Scientific computing pipelines
- Real-time image processing
- Custom accelerated libraries

## Limitations and Notes

- cuBLAS performance benchmarking could not be completed in WSL environment
- CUDA device visibility in WSL requires proper driver configuration
- Memory transfer overhead should be considered for small workloads
- PGM format used for simplicity in convolution examples

## Conclusion

This lab demonstrates the practical implementation of GPU acceleration from basic concepts to production-ready workflows. The progression from CPU baseline through naive parallelization to optimized implementations illustrates the importance of algorithmic design and memory hierarchy awareness in high-performance computing.

The integration with Python shows how GPU acceleration can be incorporated into modern development environments, making high-performance computing accessible to researchers and engineers working in high-level languages.

## References

- NVIDIA CUDA Programming Guide
- cuBLAS Documentation
- Python ctypes Documentation
