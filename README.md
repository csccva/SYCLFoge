# tinyml

A small machine-learning framework built from scratch in C++20 and SYCL.

`tinyml` is an educational project for understanding how neural-network
operations map to GPU kernels. It keeps memory management, tensor
storage, GPU operations, neural-network layers, and Transformer
components explicit rather than hiding them behind a large framework.

Development currently uses Intel oneAPI `icpx` and an Intel integrated
GPU. The immediate model target is GPT-2 Small inference. The project
has implemented and numerically validated a complete Transformer-block
forward pass at GPT-2 Small block dimensions; tokenization,
pretrained-weight loading, embeddings, stacking the full model, and
autoregressive generation remain future work.

## Current status

The following configuration has passed the integrated
`TransformerBlock::forward()` correctness test against CPU reference
calculations:

``` text
B       = 2
L       = 1024
d_model = 768
heads   = 12
d_head  = 64
FF      = 3072
dtype   = float32
```

The test validates intermediate tensors and the final block output:
input LayerNorm, Q/K/V projections, causal multi-head attention, output
projection, residual path, second LayerNorm, feed-forward projection,
GELU, second feed-forward projection, and final residual output. All
comparisons passed with an absolute tolerance of `5e-4`. In this run,
the largest reported absolute error was approximately `4.77e-7` for the
second LayerNorm and `7.45e-8` for the final output.

This is evidence of correctness for the tested configuration, not a
proof for all shapes or inputs. The test uses deterministic inputs and
weights, and the CPU reference shares the intended mathematical design.
Randomized tests, more shape coverage, and independent checks are still
useful. The test validates one Transformer block, not a complete
pretrained GPT-2 model.

## Goals

The project is intended to build understanding of GPU programming and ML
internals by implementing the components needed for a small
language-model inference pipeline.

The development path is:

1.  GPU memory management and tensor storage
2.  Basic tensor operations and matrix multiplication
3.  Neural-network layers and activations
4.  Softmax and LayerNorm
5.  Scaled dot-product and multi-head attention
6.  Causal attention and online softmax
7.  A reusable Transformer block
8.  Token and learned positional embeddings
9.  GPT-2 weight loading and model assembly
10. Tokenization and language-model output
11. Autoregressive generation
12. Further kernel optimization

Inference is the initial focus. Automatic differentiation and training
may be explored later.

## Implemented functionality

### Device memory and tensors

`DeviceBuffer<T>` provides RAII-style ownership of SYCL USM device
allocations. It uses `sycl::malloc_device`, releases owned memory
automatically, and prevents accidental shallow copies.

`Tensor<T>` provides a minimal multidimensional tensor abstraction with
shape metadata, element count, device storage, and access to the
underlying device pointer.

Example:

``` cpp
tinyml::Tensor<float> X(queue, {32, 128});
```

This creates a tensor with shape `[32,128]`. Tensor data resides in
device memory; host/device transfers are performed explicitly through
the copy operations.

### Tensor operations

GPU operations are implemented in `ops.hpp`. The current set includes:

-   Host-to-device and device-to-host copies
-   Elementwise arithmetic and addition
-   Fill and scalar multiplication
-   Bias addition
-   ReLU
-   Exact GELU and tanh-approximate GELU
-   Serial and parallel softmax
-   Serial and parallel LayerNorm
-   Affine parallel LayerNorm
-   Naive matrix multiplication
-   Tiled matrix multiplication
-   Tiled matrix multiplication with a transposed second operand
-   Scaled tiled multiplication with a transposed second operand
-   Tiled online-softmax weighted-value product from precomputed
    attention scores
-   Fused scaled dot-product attention using online softmax
-   Multi-head attention
-   Strided multi-head attention on `[B,L,d_model]` tensors
-   Causal strided multi-head attention

Operations return `sycl::event` objects where applicable. Event
dependencies express ordering between asynchronous GPU operations.

### Matrix multiplication

Several matrix-multiplication kernels are available, including naive and
tiled implementations. The tiled implementations use SYCL local memory
to reuse matrix data within work-groups.

The transposed operation computes

\[ C = AB\^T \]

for `A` with shape `[M,K]`, `B` with shape `[N,K]`, and `C` with shape
`[M,N]`. It avoids creating a physical transposed copy of `B`.

A scaled variant computes

\[ C = `\frac{AB^T}{\sqrt{K}}`{=tex}, \]

which is the score calculation used by scaled dot-product attention when
`A=Q`, `B=K`, and `K` is the attention head dimension. Local-memory
padding is used in the transposed implementation to help reduce
unfavorable bank-access patterns.

These kernels are learning and optimization exercises, not replacements
for tuned BLAS libraries.

### Linear layers

`Linear<T>` implements a fully connected layer:

\[ Y = XW + b \]

where `X` has shape `[batch_size,in_features]`, `W` has shape
`[in_features,out_features]`, `b` has shape `[out_features]`, and `Y`
has shape `[batch_size,out_features]`.

The layer also supports Transformer-style 3D tensors:

``` text
X [B,L,in_features]
        |
        v
      Linear
        |
        v
Y [B,L,out_features]
```

The contiguous `[B,L,in_features]` tensor is treated as a
`[B*L,in_features]` matrix for multiplication. No physical reshape or
data movement is required.

### Activation functions

ReLU and two GELU formulations are implemented as SYCL kernels.

ReLU:

\[ `\operatorname{ReLU}`{=tex}(x)=`\max`{=tex}(0,x) \]

Exact GELU:

\[
`\operatorname{GELU}`{=tex}(x)=`\frac{x}{2}`{=tex}`\left`{=tex}(1+`\operatorname{erf}`{=tex}`\left`{=tex}(`\frac{x}{\sqrt{2}}`{=tex}`\right`{=tex})`\right`{=tex})
\]

Tanh approximation:

\[
`\operatorname{GELU}`{=tex}(x)`\approx`{=tex}`\frac{x}{2}`{=tex}`\left[1+\tanh\left(\sqrt{\frac{2}{\pi}}(x+0.044715x^3)\right)\right]`{=tex}\]

The implementations have been compared against CPU references. Exact and
approximate GELU have also been compared for numerical accuracy and
execution time using SYCL event profiling.

### Softmax and LayerNorm

Both serial and parallel softmax implementations are available. The
parallel version uses work-group reductions to normalize rows. It is
used by the unfused attention path, while online-softmax state is used
by the tiled and fused attention kernels.

Serial and parallel LayerNorm implementations are also available. The
affine parallel version supports learned parameters:

\[
Y=`\gamma`{=tex}`\frac{X-\mu}{\sqrt{\sigma^2+\epsilon}}`{=tex}+`\beta`{=tex}
\]

It is used in the pre-LayerNorm Transformer block.

### Attention implementations

The project contains multiple attention implementations intended to make
the algorithmic and memory trade-offs visible.

#### Unfused scaled dot-product attention

The operation is

\[
`\operatorname{Attention}`{=tex}(Q,K,V)=`\operatorname{softmax}`{=tex}`\left`{=tex}(`\frac{QK^T}{\sqrt{d_k}}`{=tex}`\right`{=tex})V.
\]

The unfused path computes the score matrix with tiled `QK^T`, applies
parallel softmax, and multiplies the weights by `V`. It is a useful
reference implementation, but materializes the score and softmax-weight
matrices.

#### Tiled online-softmax weighted-value product

`softmax_matmul_tiled` computes `O = softmax(S)V` from precomputed
scores `S`. It streams over sequence tiles and maintains a running
maximum, normalization factor, and output accumulator. When a new tile
changes the running maximum, the accumulated state is rescaled to
preserve numerical consistency.

This avoids materializing the complete softmax-weight matrix, but still
requires the score matrix in global memory. It is an intermediate step
toward fully fused attention.

#### Fused attention with online softmax

The fused online-softmax implementation processes blocks of `K` and `V`
while maintaining the softmax state incrementally. For a new score tile
with maximum `m_tile`, the running maximum is updated as

\[
m\_{`\mathrm{new}`{=tex}}=`\max`{=tex}(m\_{`\mathrm{old}`{=tex}},m\_{`\mathrm{tile}`{=tex}}).
\]

The denominator and output accumulator are rescaled consistently as new
tiles are processed. This avoids storing full score and softmax-weight
matrices.

The implementation is inspired by the blockwise online-softmax approach
used by memory-efficient algorithms such as FlashAttention. It is an
educational implementation, not a reproduction of an optimized
FlashAttention kernel. In particular, the current implementation updates
the running output accumulator through global device memory; a more
optimized implementation could keep output fragments in registers or
private storage.

#### Multi-head and strided attention

The project supports an explicit head-separated representation
`[B,H,L,d_head]` and a strided representation that operates directly on
projected tensors:

``` text
Q [B,L,d_model]
K [B,L,d_model]
V [B,L,d_model]
```

The feature dimension is logically divided among heads:

\[ d\_{`\mathrm{head}`{=tex}}=`\frac{d_{\mathrm{model}}}{H}`{=tex}. \]

For head `h`, the feature offset is `h * d_head`. The strided kernel
uses the full `d_model` row stride while accessing only the features
belonging to the selected head. It therefore avoids a physical
rearrangement to `[B,H,L,d_head]`.

The implementation launches attention independently for each batch/head
pair and joins the resulting SYCL events into a completion event.

#### Causal multi-head attention

The causal kernel restricts each query at position `i` to keys and
values at positions `j <= i`. Future positions are excluded from the
softmax. It operates directly on `[B,L,d_model]` projected tensors and
retains the strided head layout.

Causal attention has been tested at component level and integrated into
the Transformer block. A useful invariant is that the first token can
attend only to itself, so its attention output equals its value vector
for each head.

#### Attention memory experiments

For square unfused attention with sequence length `N`, the score and
softmax-weight matrices each have shape `[N,N]`. With FP32, those two
matrices alone require `8*N*N` bytes.

Previous experiments on the development Intel integrated GPU ran the
unfused implementation at `N=32768`, where the two matrices together
require approximately 8 GiB, and encountered a SYCL integer-range check
at `N=49152`. The fused implementation ran through `N=65536` in the same
experiment. These are observations from a particular experiment, not
general hardware limits or performance claims.

### Transformer block

`TransformerBlock<T>` encapsulates the complete pre-LayerNorm block
forward path. It owns the attention projections and feed-forward layers
and exposes `forward()` to compose the operations through SYCL event
dependencies.

The data path is:

``` text
X [B,L,d_model]
        |
        v
     LayerNorm 1
        |
        v
   Q / K / V projections
        |
        v
  causal multi-head attention
        |
        v
    output projection
        |
        v
    residual add with X
        |
        v
     LayerNorm 2
        |
        v
Linear d_model -> d_ff
        |
        v
       GELU
        |
        v
Linear d_ff -> d_model
        |
        v
 residual add
        |
        v
       Output
```

The block has been validated against CPU reference calculations at the
following dimensions:

``` text
B       = 2
L       = 1024
d_model = 768
heads   = 12
d_head  = 64
d_ff    = 3072
```

The test compares intermediate results as well as the final output. In
the reported run, all comparisons passed with absolute tolerance `5e-4`;
the largest absolute error was approximately `4.77e-7` for the second
LayerNorm and `7.45e-8` for the final output. The inputs and weights in
this test are deterministic. Broader randomized and multi-shape testing
remains part of strengthening validation.

This validates a single block at GPT-2 Small dimensions. It does not
mean that the complete GPT-2 model is implemented.

## Testing and benchmarks

Correctness tests are separated from performance experiments. The
current test layout is:

``` text
tests/
├── test_core.cpp
├── test_nn.cpp
├── test_sdpa.cpp
├── test_attention.cpp
├── test_flash_attention.cpp
└── benchmark_ops.cpp
```

The intended responsibilities are:

-   `test_core.cpp` --- tensor operations and matrix multiplication
-   `test_nn.cpp` --- activations, softmax, LayerNorm, Linear, and
    feed-forward/MLP operations
-   `test_sdpa.cpp` --- scaled dot-product attention and fused
    online-softmax attention
-   `test_attention.cpp` --- multi-head, strided, causal, and
    Transformer-block integration tests
-   `test_flash_attention.cpp` --- CPU-reference comparisons for tiled
    attention operations, including online-softmax weighted-value
    products
-   `benchmark_ops.cpp` --- performance experiments, including GELU
    variants, tiled matrix multiplication, fused versus unfused
    attention, and sequence-length scaling

The exact contents of test executables can evolve as the project
develops. Keep correctness checks and performance benchmarks separate so
performance experiments do not obscure validation results.

## GPT-2 Small target

GPT-2 Small is the first intended pretrained-model target. Its standard
architecture is:

``` text
vocabulary size       = 50257
context length        = 1024
d_model               = 768
attention heads       = 12
d_head                = 64
Transformer blocks    = 12
feed-forward dimension= 3072
```

The computational operations for one GPT-2-style block are implemented
and the block forward pass has passed a CPU-reference comparison at
these dimensions. The complete model still requires token embeddings,
learned positional embeddings, assembly of 12 blocks, final LayerNorm,
loading pretrained weights, the language-model head, and
tokenization/generation integration.

## Building

A SYCL-capable C++ compiler is required. Development currently uses
Intel oneAPI `icpx`.

Configure with tests enabled:

``` bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=icpx \
  -DTINYML_BUILD_TESTS=ON
```

Build the main executable:

``` bash
cmake --build build --target tinyml
```

Build correctness tests:

``` bash
cmake --build build --target test_core test_nn test_sdpa test_attention test_flash_attention
```

Build the benchmark executable:

``` bash
cmake --build build --target benchmark_ops
```

Run the executables:

``` bash
./build/test_core
./build/test_nn
./build/test_sdpa
./build/test_attention
./build/test_flash_attention
./build/benchmark_ops
```

The target names above reflect the project layout described here; check
`CMakeLists.txt` if a target differs in your current checkout.

## Design principles

-   C++20
-   SYCL for GPU programming
-   USM device memory
-   Explicit `nd_range` kernels where appropriate
-   Asynchronous operations linked through `sycl::event`
-   Minimal abstractions
-   GPU operations separated from neural-network layers
-   Correctness before optimization
-   CPU references used to validate GPU results

The purpose is to make the path from a neural-network operation to the
underlying SYCL kernel understandable.

## Project structure

``` text
tinyml/
├── CMakeLists.txt
├── LICENSE
├── README.md
├── include/
│   └── tinyml/
│       ├── device_buffer.hpp
│       ├── tensor.hpp
│       ├── ops.hpp
│       ├── flash_attention.hpp
│       ├── linear.hpp
│       └── transformer_block.hpp
├── src/
│   └── main.cpp
├── examples/
├── tests/
│   ├── test_core.cpp
│   ├── test_nn.cpp
│   ├── test_sdpa.cpp
│   ├── test_attention.cpp
│   ├── test_flash_attention.cpp
│   └── benchmark_ops.cpp
└── build/
```

This is a representative structure; adjust it to match the files present
in the current checkout.

## Roadmap

The next steps are:

-   Add randomized inputs and weights with multiple seeds to
    Transformer-block correctness tests
-   Expand shape coverage across batch size, sequence length, embedding
    dimension, head count, and feed-forward width
-   Add token embeddings and learned positional embeddings
-   Add final model LayerNorm and GPT-2 weight loading
-   Assemble the 12-block GPT-2 Small model
-   Add the language-model head and tied token-embedding weights
-   Integrate tokenization
-   Implement autoregressive generation
-   Profile and optimize bottlenecks after correctness is established
-   Continue attention-kernel work: compute `QK^T` tiles on the fly,
    avoid materializing the score matrix, keep output fragments in
    registers/private storage, and tune work-group mapping and
    synchronization
-   Explore a Triton version of the attention algorithm to compare its
    programming model with SYCL
-   Explore portability to another SYCL implementation such as
    AdaptiveCpp
-   Consider automatic differentiation and training as a longer-term
    extension

## Status summary

Implemented and tested components include USM device memory management,
multidimensional tensors, host/device transfers, elementwise operations,
naive and tiled matrix multiplication, linear layers, ReLU and GELU,
serial and parallel softmax, affine parallel LayerNorm, unfused scaled
dot-product attention, tiled online-softmax weighted-value products,
fused online-softmax attention, multi-head and strided attention, causal
attention, Q/K/V projections, attention output projection, residual
connections, the Transformer feed-forward path, and a reusable
`TransformerBlock<T>` forward pass.

The current milestone is a validated single Transformer block at GPT-2
Small dimensions. The next major milestone is a complete pretrained
GPT-2 Small inference path, including embeddings, model weights,
tokenization, and autoregressive generation.
