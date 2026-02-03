# CUDA Matrix Multiplication Project
---

## Overview

This project explores matrix multiplication performance across multiple computational models, starting from a basic CPU implementation and progressively moving toward GPU acceleration using CUDA. The goal is to understand:

- How matrix multiplication is formulated computationally
- How performance scales with matrix size
- How GPU optimizations improve performance
- How highly optimized libraries (cuBLAS) compare to custom kernels
- How CUDA code can be exposed as a reusable Python library

The project is structured as a gradual evolution of the same mathematical problem, implemented using increasingly advanced techniques.

---

## Problem Formulation

Given two square matrices **A** and **B** of size `N × N`, compute the output matrix **C** such that:

C[i][j] = Σ (A[i][k] × B[k][j]) for k = 0 to N−1


This formulation is identical across all implementations. What changes is **where and how the computation is executed**.

---

## Development Environment

### Hardware & Software

- OS: Windows with WSL (Ubuntu)
- Compiler: `gcc` (CPU), `nvcc` (CUDA)
- CUDA Toolkit: CUDA 13
- GPU: NVIDIA GPU (optional; code supports fallback if not detected)
- Libraries:
  - CUDA Runtime
  - cuBLAS
  - Python 3
  - ctypes (Python FFI)

> Note: On systems without CUDA-capable GPUs (or without GPU passthrough in WSL), CUDA programs compile successfully but skip execution gracefully.

---

## Project Structure


    ├── cpu_matmul.c 
    ├── matmul_naive.cu
    ├── matmul_tiled.cu
    ├── matmul_cublas.cu
    ├── matmul_lib.cu
    ├── matmul.py
    └── README.md


---

## Part 1: CPU Matrix Multiplication (Baseline)

### Objective
Establish a baseline implementation to compare GPU speedups.

### Approach
- Nested loops iterate over rows, columns, and the shared dimension.
- All computation occurs sequentially on the CPU.

### Key Characteristics
- Simple and easy to understand
- Poor performance for large matrices (`O(N³)` operations)
- Serves as correctness reference

---

## Part 2: Naïve CUDA Matrix Multiplication

### Objective
Introduce GPU parallelism using CUDA.

### Core Idea
- Each CUDA thread computes **one element** of matrix `C`.
- Threads are organized in a 2D grid of blocks.

### Kernel Logic
1. Compute global row and column indices using `blockIdx`, `blockDim`, and `threadIdx`
2. Perform the dot product for that `(row, col)`
3. Write the result to global memory

### Characteristics
- Massive parallelism
- High global memory access cost
- Minimal optimization

---

## Part 3: Performance Measurement

### Methodology
- Use CUDA events (`cudaEventRecord`) to measure kernel execution time
- Test matrix sizes:
  - 512 × 512
  - 1024 × 1024
  - 2048 × 2048

This provides a fair comparison between CPU and GPU implementations.

---

## Part 4: Optimized CUDA with Shared Memory Tiling

### Objective
Reduce global memory access and improve performance.

### Optimization Strategy: Tiling

- Matrices are divided into `TILE_WIDTH × TILE_WIDTH` tiles
- Tiles are loaded into **shared memory**
- Threads reuse data within a tile

### Kernel Flow
1. Load tiles of `A` and `B` into shared memory
2. Synchronize threads
3. Compute partial dot products
4. Accumulate results across tiles
5. Write final value to global memory

### Benefits
- Fewer global memory accesses
- Better cache utilization
- Significant performance improvement over naïve CUDA

---

## Part 5: Performance Comparison

All implementations were benchmarked using identical matrix sizes.

| Implementation     | N=512 | N=1024 | N=2048 |
|--------------------|-------|--------|--------|
| CPU (C)            | X sec | Y sec  | Z sec  |
| Naïve CUDA         | A ms  | B ms   | C ms   |
| Optimized CUDA     | D ms  | E ms   | F ms   |
| cuBLAS             | G ms  | H ms   | I ms   |

---

## Part 6: cuBLAS Matrix Multiplication

### Objective
Use NVIDIA’s highly optimized BLAS library for maximum performance.

### Implementation
- Uses `cublasSgemm`
- Leverages hardware-specific optimizations
- Uses column-major layout (handled by swapping operands)

### Advantages
- Industry-grade performance
- Minimal code
- Often outperforms custom kernels

### GPU Detection
The program checks for a CUDA-capable GPU at runtime and exits gracefully if none is found.

---

## Part 7: Creating a CUDA Shared Library

### Objective
Expose CUDA functionality as a reusable library.

### Steps
1. Wrap CUDA kernel in `extern "C"` function
2. Compile as a shared library (`.so`)
3. Ensure clean memory allocation and synchronization

This allows non-C++ programs to call CUDA code.

---

## Part 8: Python Integration Using ctypes

### Objective
Use CUDA matrix multiplication directly from Python.

### Approach
- Load the shared library using `ctypes.CDLL`
- Define argument and return types
- Pass NumPy arrays to CUDA functions

### Benefits
- Combines Python ease-of-use with CUDA performance
- Enables rapid experimentation
- Useful for ML and scientific computing workflows

---

## Key Takeaways

- Matrix multiplication performance improves dramatically with parallelism
- Memory access patterns matter more than raw computation
- Shared memory tiling is critical for CUDA performance
- cuBLAS provides near-peak GPU efficiency
- CUDA code can be cleanly integrated into Python workflows

---

## Conclusion

This project demonstrates a full performance engineering pipeline:

**CPU → GPU → Optimized GPU → Library → Python API**

By building each layer step-by-step, we gain both theoretical understanding and practical experience with high-performance computing on modern hardware.

---
---