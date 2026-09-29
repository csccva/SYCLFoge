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
- serial and parallel softmax
- serial and parallel layer normalization
- matrix multiplication
- tiled matrix multiplication
- tiled multiplication with a transposed second operand
- scaled tiled multiplication with a transposed second operand
- fused scaled dot-product attention using online softmax

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

The parallel softmax is used by the unfused scaled dot-product attention implementation. A separate online-softmax formulation is used by the fused attention kernel.

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

### Scaled dot-product attention

Single-head scaled dot-product attention is implemented and tested.

The operation is:

\[
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

The initial implementation composes three existing GPU operations:

```text
Q ──┐
    ├── QK^T / sqrt(dk) ──> scores ──> softmax ──> weights ──┐
K ──┘                                                        ├──> output
V ───────────────────────────────────────────────────────────┘
```

The scaled score calculation uses the tiled transposed matrix-multiplication kernel, so an explicit transposed copy of `K` is not required.

The resulting score matrix is normalized using the parallel softmax kernel and then multiplied by `V` using tiled matrix multiplication.

This implementation provides a simple reference for exploring attention before kernel fusion.

### Fused attention with online softmax

A fused attention implementation has also been developed.

Instead of materializing the complete score and softmax-weight matrices, the kernel processes blocks of `K` and `V` while maintaining the softmax state incrementally.

For each query row, the kernel maintains:

- a running maximum \(m\)
- a running softmax denominator \(D\)
- a running unnormalized output \(U\)

For a new score tile with maximum \(m_{\mathrm{tile}}\), the running maximum becomes:

\[
m_{\mathrm{new}} = \max(m_{\mathrm{old}},m_{\mathrm{tile}})
\]

and the denominator is updated using:

\[
D_{\mathrm{new}}
=
D_{\mathrm{old}}e^{m_{\mathrm{old}}-m_{\mathrm{new}}}
+
D_{\mathrm{tile}}e^{m_{\mathrm{tile}}-m_{\mathrm{new}}}
\]

The output numerator is rescaled in the same way while accumulating the contribution from the corresponding `V` tile.

After all key/value tiles have been processed, the accumulated output is normalized by the final denominator.

This allows attention to be evaluated without storing the full:

```text
scores  [M, N]
weights [M, N]
```

intermediate tensors.

For fixed attention dimension \(d_k\), this changes the intermediate-memory behavior from quadratic in sequence length to a blockwise streaming approach.

The fused implementation has been validated numerically against the unfused attention implementation over multiple K/V tiles. For one larger correctness test with `M=127`, `N=259`, and `K=128`, the maximum difference between the implementations was approximately `6.2e-9`.

The implementation is inspired by the online-softmax/blockwise approach used by memory-efficient attention algorithms such as FlashAttention, but it is an educational implementation rather than a reproduction of an optimized FlashAttention kernel.

In particular, the current implementation updates the running output accumulator through global device memory. A more optimized implementation would keep output fragments in per-thread private/register storage while streaming K/V blocks.

### Attention memory scaling

The unfused and fused implementations have also been tested with increasing sequence lengths.

For square attention with `M=N`, the unfused implementation requires two large intermediate matrices:

\[
\mathrm{scores},\mathrm{weights} \in \mathbb{R}^{N\times N}
\]

For FP32, these two matrices alone require:

\[
8N^2 \text{ bytes}
\]

The unfused implementation successfully ran at `N=32768`, where the score and weight matrices together require approximately 8 GiB.

The next tested size, `N=49152`, encountered the default SYCL integer-range check rather than an out-of-memory failure.

The fused implementation, which does not allocate these quadratic intermediates, successfully ran through `N=65536` in the same experiment.

These tests demonstrate the different memory-scaling behavior of the two approaches. They should not be interpreted as representative performance measurements of dedicated GPU hardware.

Current development and benchmarking are performed on an Intel integrated GPU that also drives the operating-system display, so performance measurements can be affected by other GPU activity and shared-memory behavior.

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

- multi-head attention
- embeddings
- transformer blocks
- token and weight loading
- tiny Transformer inference
- further GPU kernel optimization

Attention optimization can later explore:

- register/private-memory output accumulation
- improved work-group and output-fragment mapping
- reduced synchronization and local-memory usage
- architecture-specific tuning

A parallel learning path is to implement the attention algorithm in Triton and compare its block and register programming model with the SYCL implementation.

Longer term, the project may explore automatic differentiation and training.

## Status

Work in progress.

The project currently supports an end-to-end GPU inference pipeline composed from tensors, custom SYCL kernels, linear layers, activation functions, softmax, layer normalization, feed-forward networks, and single-head scaled dot-product attention.

Both unfused and fused attention implementations are now available. The unfused implementation composes tiled \(QK^T/\sqrt{d_k}\), parallel softmax, and tiled multiplication with `V`. The fused implementation uses blockwise processing and online softmax to avoid materializing the full attention score and weight matrices.

The fused kernel has been validated against the unfused implementation and used to explore the difference between conventional attention's quadratic intermediate-memory requirements and blockwise attention.

The next major development stage is multi-head attention, followed by the remaining components required for a small Transformer inference pipeline.

This is an educational project for exploring C++, SYCL, GPU programming, GPU memory-access patterns, parallel reductions, kernel fusion, online softmax, and the internals of machine-learning frameworks. The kernels are intentionally implemented directly rather than delegating the work to optimized ML or BLAS libraries.