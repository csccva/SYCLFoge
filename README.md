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
7. Attention
8. Multi-head attention
9. Transformer building blocks
10. Transformer model
11. Tiny LLM inference

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
- multi-head attention
- strided multi-head attention over `[B,L,d_model]` tensors

Operations return `sycl::event` objects so dependencies between asynchronous GPU operations can be expressed explicitly.

### Matrix multiplication

Several matrix multiplication operations are currently available:

- naïve matrix multiplication
- tiled matrix multiplication using SYCL local memory
- tiled `AB^T` matrix multiplication
- scaled tiled `AB^T` matrix multiplication for attention

The transposed operation computes:

**C = AB^T**

for:

- `A` with shape `[M,K]`
- `B` with shape `[N,K]`
- `C` with shape `[M,N]`

This avoids explicitly constructing a transposed copy of `B`.

A scaled version computes:

**C = AB^T / sqrt(K)**

which provides the score computation required by scaled dot-product attention when `A = Q`, `B = K`, and the matrix dimension `K` corresponds to the attention dimension `d_k`.

The tiled implementations use SYCL local memory to reuse matrix data within work-groups. Local-memory padding is used in the transposed implementation to reduce unfavorable strided bank-access patterns.

The matrix multiplication kernels are intended primarily for learning GPU optimization techniques. They are not intended to replace optimized BLAS implementations.

### Linear layer

A basic fully connected layer is implemented:

**Y = XW + b**

where:

- `X` has shape `[batch_size,in_features]`
- `W` has shape `[in_features,out_features]`
- `b` has shape `[out_features]`
- `Y` has shape `[batch_size,out_features]`

The matrix multiplication and bias kernels are connected through SYCL event dependencies.

The linear layer also supports 3D Transformer-style tensors:

```text
X [B,L,in_features]
        |
        v
Linear
        |
        v
Y [B,L,out_features]
```

The `[B,L,in_features]` tensor is interpreted as a contiguous `[B*L,in_features]` matrix during matrix multiplication, so no physical reshape or data movement is required.

### Activation functions

ReLU and GELU activation functions are implemented as SYCL GPU kernels.

ReLU:

**ReLU(x) = max(0,x)**

Two GELU implementations are available:

- exact GELU using the error function
- approximate GELU using the tanh approximation

The exact formulation is:

**GELU(x) = x / 2 * (1 + erf(x / sqrt(2)))**

The approximate formulation is:

**GELU(x) ≈ x / 2 * [1 + tanh(sqrt(2 / pi) * (x + 0.044715x^3))]**

Both implementations have been tested against CPU references. The exact and approximate implementations have also been compared for numerical accuracy and GPU execution time using SYCL event profiling.

### Softmax

Both serial and parallel GPU implementations of softmax are available.

The parallel implementation uses work-group reductions to compute the normalization across rows and serves as an exercise in implementing reduction-based neural-network operations with SYCL.

The parallel softmax is used by the unfused scaled dot-product attention implementation. A separate online-softmax formulation is used by the fused attention kernel.

### Layer normalization

Both serial and parallel implementations of layer normalization are available.

The parallel implementation uses SYCL work-group reductions to compute row statistics and supports learned affine parameters:

**Y = gamma * (X - mean) / sqrt(variance + eps) + beta**

The affine parallel implementation is used for the pre-LayerNorm structure of the GPT-2 Transformer block.

The implementation is primarily intended for learning parallel reduction techniques and floating-point behavior on GPUs.

### Feed-forward network

The implemented components can be composed into a small neural-network inference pipeline:

```text
X [batch,2]
    |
    v
Linear(2,4)
    |
    v
ReLU / GELU
    |
    v
Linear(4,2)
    |
    v
Y [batch,2]
```

The neural-network operations are executed on the GPU, with input data and parameters explicitly transferred to device memory.

Dependencies between parameter transfers and successive operations are expressed using `sycl::event` objects, allowing kernels to be chained without requiring host synchronization between layers.

### Scaled dot-product attention

Single-head scaled dot-product attention is implemented and tested.

The operation is:

**Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V**

The initial implementation composes three existing GPU operations:

```text
Q ──┐
    ├── QK^T / sqrt(d_k) ──> scores ──> softmax ──> weights ──┐
K ──┘                                                         ├──> output
V ────────────────────────────────────────────────────────────┘
```

The scaled score calculation uses the tiled transposed matrix-multiplication kernel, so an explicit transposed copy of `K` is not required.

The resulting score matrix is normalized using the parallel softmax kernel and then multiplied by `V` using tiled matrix multiplication.

This implementation provides a simple reference for exploring attention before kernel fusion.

### GPT-2 feed-forward network

The GPT-2-style Transformer feed-forward path has been implemented using the existing 3D linear layer and GELU operation:

```text
X [B,L,768]
        |
        v
Linear 768 -> 3072
        |
        v
      GELU
        |
        v
Linear 3072 -> 768
        |
        v
Y [B,L,768]
```

### Fused attention with online softmax

A fused attention implementation has also been developed.

Instead of materializing the complete score and softmax-weight matrices, the kernel processes blocks of `K` and `V` while maintaining the softmax state incrementally.

For each query row, the kernel maintains:

- a running maximum `m`
- a running softmax denominator `D`
- a running unnormalized output `U`

For a new score tile with maximum `m_tile`, the running maximum becomes:

**m_new = max(m_old,m_tile)**

and the denominator is updated using:

**D_new = D_old * exp(m_old - m_new) + D_tile * exp(m_tile - m_new)**

The output numerator is rescaled in the same way while accumulating the contribution from the corresponding `V` tile.

After all key/value tiles have been processed, the accumulated output is normalized by the final denominator.

This allows attention to be evaluated without storing the full:

```text
scores  [M,N]
weights [M,N]
```

intermediate tensors.

For fixed attention dimension `d_k`, this changes the intermediate-memory behavior from quadratic in sequence length to a blockwise streaming approach.

The fused implementation has been validated numerically against the unfused attention implementation over multiple K/V tiles. For one larger correctness test with `M=127`, `N=259`, and `K=128`, the maximum difference between the implementations was approximately `6.2e-9`.

The implementation is inspired by the online-softmax/blockwise approach used by memory-efficient attention algorithms such as FlashAttention, but it is an educational implementation rather than a reproduction of an optimized FlashAttention kernel.

In particular, the current implementation updates the running output accumulator through global device memory. A more optimized implementation would keep output fragments in per-thread private/register storage while streaming K/V blocks.

### Multi-head attention

Multi-head attention is implemented using the fused online-softmax attention kernel.

An initial implementation operates on tensors where the heads have already been physically separated:

```text
[B,H,L,d_head]
```

This implementation is retained as a useful reference.

A second implementation operates directly on Transformer-style projected tensors:

```text
Q [B,L,d_model]
K [B,L,d_model]
V [B,L,d_model]
```

The model dimension is divided into attention heads:

**d_head = d_model / H**

where `H` is the number of heads.

Logically, the feature dimension is interpreted as:

```text
d_model

| head 0 | head 1 | head 2 | ... |
```

Rather than physically rearranging the tensors into `[B,H,L,d_head]`, the strided attention kernel accesses each head directly inside the original `[B,L,d_model]` memory layout.

For head `h`, the starting feature offset is:

```text
j_start = h * d_head
```

The attention kernel then uses the full `d_model` row stride while operating only on the `d_head` features belonging to that head.

For each batch and head, the effective operation is therefore:

```text
Q[b,:,head] [L,d_head]
K[b,:,head] [L,d_head]
V[b,:,head] [L,d_head]
             |
             v
      fused attention
             |
             v
U[b,:,head] [L,d_head]
```

while the physical tensors remain:

```text
[B,L,d_model]
```

This avoids an intermediate physical transpose or rearrangement from `[B,L,d_model]` to `[B,H,L,d_head]`.

The implementation launches attention independently for each `(batch,head)` pair. The resulting SYCL events are collected and joined into a single completion event.

The implementation has been tested at several levels:

- the strided single-head kernel against the contiguous single-head implementation
- the multi-head strided wrapper against individual head calls
- the batched multi-head strided implementation against explicit per-batch, per-head execution

The current batched correctness test uses `B=4`, `L=4`, `d_model=8`, and two attention heads and produces identical output to the explicit reference execution.

### Causal multi-head attention

A separate causal version of the strided fused attention kernel is implemented for autoregressive Transformer inference.

For query position `i`, only key/value positions satisfying:

**j <= i**

participate in attention. Future positions are masked from the softmax.

The causal implementation operates directly on projected `[B,L,d_model]` tensors and preserves the same strided head layout as the non-causal implementation.

The causal path has been tested for individual heads and for batched multi-head attention. It has also been integrated with learned Q/K/V projections and tested at GPT-2 Small dimensions with `d_model=768` and `12` attention heads.

### Attention memory scaling

The unfused and fused implementations have also been tested with increasing sequence lengths.

For square attention with `M=N`, the unfused implementation requires two large intermediate matrices:

```text
scores  [N,N]
weights [N,N]
```

For FP32, these two matrices alone require:

**8N^2 bytes**

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

- Transformer block abstraction
- token and positional embeddings
- final layer normalization
- GPT-2 weight loading
- language-model head with tied token-embedding weights
- tokenization integration
- GPT-2 Small inference
- autoregressive generation
- test reorganization and optional test builds
- further GPU kernel optimization

The first complete pretrained model target is GPT-2 Small.

Attention optimization can later explore:

- register/private-memory output accumulation
- improved work-group and output-fragment mapping
- reduced synchronization and local-memory usage
- architecture-specific tuning

A parallel learning path is to implement the attention algorithm in Triton and compare its block and register programming model with the SYCL implementation.

After the SYCL implementation is sufficiently complete, portability to another SYCL implementation such as AdaptiveCpp can also be explored.

Longer term, the project may explore automatic differentiation and training.

## Status

Work in progress.

The project now contains the main computational components required for a GPT-2 Transformer block.

Implemented components include:

- USM-based device memory management
- multidimensional tensor storage
- host/device transfers
- elementwise tensor operations
- naïve and tiled matrix multiplication
- linear layers
- 3D linear layers for `[B,L,D]` tensors
- ReLU and GELU
- serial and parallel softmax
- affine parallel layer normalization
- unfused scaled dot-product attention
- fused attention using blockwise online softmax
- multi-head attention
- strided batched multi-head attention operating directly on `[B,L,d_model]`
- causal strided multi-head attention
- Q/K/V projections
- attention output projection
- Transformer residual connections
- GPT-2-style feed-forward network

The current Transformer block data path is:

```text
X [B,L,d_model]
        |
        v
   LayerNorm 1
        |
        v
   Q/K/V projections
        |
        v
causal multi-head attention
        |
        v
       W_O
        |
        v
   residual + X
        |
        v
        R
        |
        v
   LayerNorm 2
        |
        v
Linear d_model -> 4*d_model
        |
        v
      GELU
        |
        v
Linear 4*d_model -> d_model
        |
        v
   residual + R
        |
        v
      Output
```

