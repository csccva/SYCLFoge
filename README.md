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
- scalar multiplication
- bias addition
- ReLU
- exact GELU
- approximate GELU
- softmax
- layer normalization
- matrix multiplication
- tiled matrix multiplication
- tiled multiplication with a transposed second operand
- scaled tiled multiplication with a transposed second operand

Operations return `sycl::event` objects so dependencies between asynchronous GPU operations can be expressed explicitly.

### Matrix multiplication

Several matrix multiplication operations are currently available:

- naïve matrix multiplication
- tiled matrix multiplication using SYCL local memory
- tiled `AB^T` matrix multiplication
- scaled tiled `AB^T` matrix multiplication for attention

The transposed operation computes:

\[
C = AB^T
\]

for:

- `A` with shape `[M, K]`
- `B` with shape `[N, K]`
- `C` with shape `[M, N]`

This avoids explicitly constructing a transposed copy of `B`.

A scaled version computes:

\[
C = \frac{AB^T}{\sqrt{K}}
\]

which provides the score computation required by scaled dot-product attention when `A = Q`, `B = K`, and the matrix dimension `K` corresponds to the attention dimension \(d_k\).

The tiled implementations use SYCL local memory to reuse matrix data within work-groups. Local-memory padding is used in the transposed implementation to reduce unfavorable strided bank-access patterns.

The matrix multiplication kernels are intended primarily for learning GPU optimization techniques. They are not intended to replace optimized BLAS implementations.

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

ReLU and GELU activation functions are implemented as SYCL GPU kernels.

ReLU:

\[
\mathrm{ReLU}(x) = \max(0,x)
\]

Two GELU implementations are available:

- exact GELU using the error function
- approximate GELU using the tanh approximation

The exact formulation is:

\[
\mathrm{GELU}(x) =
\frac{x}{2}
\left(
1 + \mathrm{erf}\left(\frac{x}{\sqrt{2}}\right)
\right)
\]

The approximate formulation is:

\[
\mathrm{GELU}(x) \approx
\frac{x}{2}
\left[
1 +
\tanh\left(
\sqrt{\frac{2}{\pi}}
\left(x + 0.044715x^3\right)
\right)
\right]
\]

Both implementations have been tested against CPU references. The exact and approximate implementations have also been compared for numerical accuracy and GPU execution time using SYCL event profiling.

### Softmax

Both serial and parallel GPU implementations of softmax are available.

The parallel implementation uses work-group reductions to compute the normalization across rows and serves as an exercise in implementing reduction-based neural-network operations with SYCL.

This operation will be used to normalize the scaled \(QK^T\) scores in scaled dot-product attention.

### Layer normalization

Both serial and parallel implementations of layer normalization are available.

The parallel implementation uses SYCL work-group reductions to compute row statistics. The implementation is primarily intended for learning parallel reduction techniques and floating-point behavior on GPUs.

### Feed-forward network

The implemented components can be composed into a small neural-network inference pipeline:

```text
X [batch, 2]
    |
    v
Linear(2, 4)
    |
    v
ReLU / GELU
    |
    v
Linear(4, 2)
    |
    v
Y [batch, 2]
```

The neural-network operations are executed on the GPU, with input data and parameters explicitly transferred to device memory.

Dependencies between parameter transfers and successive operations are expressed using `sycl::event` objects, allowing kernels to be chained without requiring host synchronization between layers.

### Attention primitives

Development of scaled dot-product attention is in progress.

The target operation is:

\[
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

The first stage is already implemented as a fused scaled transposed matrix multiplication:

```text
Q [Lq, dk] ──┐
             ├── scaled QK^T ──> scores [Lq, Lk]
K [Lk, dk] ──┘
```

The implementation computes:

\[
\mathrm{scores} = \frac{QK^T}{\sqrt{d_k}}
\]

without explicitly transposing `K` and without requiring a separate scaling kernel.

The scaled operation has been validated against a CPU reference. The next step is to connect it to the existing parallel softmax implementation, followed by multiplication of the resulting attention weights with `V`.

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

- complete single-head scaled dot-product attention
- multi-head attention
- embeddings
- transformer blocks
- token/weight loading
- tiny Transformer inference
- further GPU kernel optimization

Possible later optimization work includes fusing attention operations and exploring blockwise/online softmax approaches to reduce intermediate global-memory traffic.

Longer term, the project may explore automatic differentiation and training.

## Status

Work in progress.

The project currently supports a basic end-to-end GPU inference pipeline composed from tensors, custom SYCL kernels, linear layers, activation functions, softmax, and layer normalization.

Development has now reached the attention stage. Tiled `AB^T` multiplication and scaled \(QK^T/\sqrt{d_k}\) are implemented and tested, providing the first stage required for scaled dot-product attention. The next step is to connect the scaled score calculation with the existing parallel softmax implementation and then multiply the attention weights by `V`.

This is an educational project for exploring C++, SYCL, GPU programming, GPU memory-access patterns, parallel reductions, and the internals of machine-learning frameworks. The kernels are intentionally implemented directly rather than delegating the work to optimized ML or BLAS libraries.