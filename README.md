# tinyml

A small machine-learning framework built from scratch in C++ and SYCL.

The goal of this project is educational: to understand how the core components of a neural-network framework can be implemented directly on a GPU, starting from memory management and tensor operations and eventually building a small Transformer/LLM inference pipeline.

The project deliberately uses minimal abstractions so that the path from a neural-network operation to the underlying SYCL GPU kernel remains visible.

## Goals

The long-term goal is to build a small Transformer capable of language-model inference using components implemented in this project.

The development path is roughly:

1. GPU memory management
2. Tensor abstraction
3. Basic tensor operations
4. Matrix multiplication
5. Neural-network layers
6. Small feed-forward networks
7. Transformer building blocks
8. Attention
9. Transformer model
10. Tiny LLM inference

The initial focus is inference. The design may later be extended toward gradients and training.

## Current functionality

### Device memory

`DeviceBuffer<T>` provides basic RAII-based management of SYCL USM device memory.

It handles:

- device allocation with `sycl::malloc_device`
- device memory ownership
- automatic deallocation
- prevention of accidental shallow copies

### Tensor

`Tensor<T>` provides a minimal tensor abstraction containing:

- tensor shape
- number of elements
- device storage
- access to the underlying device pointer

For example:

```cpp
tinyml::Tensor<float> X(queue, {32, 128});
```

creates a tensor representing 32 samples with 128 features each.

### Tensor operations

Basic GPU operations are implemented in `ops.hpp`.

Current operations include:

- host/device copies
- elementwise arithmetic
- fill
- bias addition
- ReLU
- matrix multiplication

Operations return `sycl::event` objects so dependencies between asynchronous GPU operations can be expressed explicitly.

### Matrix multiplication

Two matrix multiplication implementations are currently available:

- naïve matrix multiplication
- tiled matrix multiplication using SYCL local memory

The tiled implementation is intended primarily for learning GPU optimization techniques. It is not intended to replace optimized BLAS implementations.

### Linear layer

A basic fully connected layer is implemented:

\[
Y = XW + b
\]

where:

- `X` has shape `[batch_size, in_features]`
- `W` has shape `[in_features, out_features]`
- `b` has shape `[out_features]`
- `Y` has shape `[batch_size, out_features]`

The matrix multiplication and bias kernels are connected through SYCL event dependencies.

### Activation functions

ReLU is currently implemented as a SYCL GPU kernel:

\[
\mathrm{ReLU}(x) = \max(0,x)
\]

### Feed-forward network

The implemented components can be composed into a small neural-network inference pipeline:

```text
X [batch, 2]
    |
    v
Linear(2, 4)
    |
    v
ReLU
    |
    v
Linear(4, 2)
    |
    v
Y [batch, 2]
```

The neural-network operations are executed on the GPU, with input data and parameters explicitly transferred to device memory.

Dependencies between parameter transfers and successive operations are expressed using `sycl::event` objects, allowing kernels to be chained without requiring host synchronization between layers.

## Design principles

The project currently follows a few simple rules:

- C++20
- SYCL for GPU programming
- USM device memory
- explicit `nd_range` kernels
- asynchronous operations using `sycl::event`
- minimal abstractions
- GPU operations kept separate from neural-network layers
- correctness before optimization

The intention is to understand the implementation rather than hide it behind a large framework.

## Project structure

```text
tinyml/
├── CMakeLists.txt
├── LICENSE
├── README.md
├── include/
│   └── tinyml/
│       ├── device_buffer.hpp
│       ├── tensor.hpp
│       ├── ops.hpp
│       └── linear.hpp
├── src/
│   └── main.cpp
├── examples/
├── tests/
└── build/
```

## Building

A SYCL-capable C++ compiler is required.

The project is currently developed using Intel oneAPI `icpx`.

```bash
cmake -S . -B build -DCMAKE_CXX_COMPILER=icpx
cmake --build build
```

Run with:

```bash
./build/tinyml
```

## Roadmap

The next major steps are:

- additional activation functions
- softmax
- embeddings
- normalization
- attention
- transformer blocks
- token/weight loading
- tiny Transformer inference
- further GPU kernel optimization

Longer term, the project may explore automatic differentiation and training.

## Status

Work in progress.

The project currently supports a basic end-to-end GPU inference pipeline composed from tensors, custom SYCL kernels, linear layers, and activation functions.

This is an educational project for exploring C++, SYCL, GPU programming, and the internals of machine-learning frameworks. The kernels are intentionally implemented directly rather than delegating the work to optimized ML or BLAS libraries.