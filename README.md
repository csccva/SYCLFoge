
# tinyml

A small machine-learning framework built from scratch in C++20 and SYCL.

`tinyml` is an educational project for understanding how neural-network operations map to GPU kernels. It keeps memory management, tensor storage, GPU operations, neural-network layers, and Transformer components explicit rather than hiding them behind a large framework.

Development currently uses Intel oneAPI `icpx` and an Intel integrated GPU. The immediate model target is GPT-2 Small inference. A single Transformer block has passed numerical validation at GPT-2 Small dimensions. The GPT-2 tokenizer wrapper and token embedding layer have also passed their initial tests. The complete pretrained model and generation pipeline remain under development.

## Current status

The project has three validated milestones:

- **Transformer block:** the complete forward pass of one pre-LayerNorm Transformer block passes comparisons against CPU reference calculations at GPT-2 Small dimensions.
- **GPT-2 tokenizer:** `GPT2Tokenizer` loads the GPT-2 `tokenizer.json` file and passes encode/decode round-trip tests for ordinary text, repeated spaces, newlines, and Unicode.
- **Token embeddings:** `Embedding<T>` passes a numerical test using known token IDs and a manually initialized embedding table.

These components are not yet assembled into a complete pretrained GPT-2 model. The embedding table is not yet loaded from pretrained GPT-2 weights, and learned positional embeddings, full model assembly, final LayerNorm, the language-model head, and autoregressive generation remain unfinished.

### Transformer-block validation

The following configuration has passed the integrated `TransformerBlock::forward()` correctness test against CPU reference calculations:

```text
B       = 2
L       = 1024
d_model = 768
heads   = 12
d_head  = 64
FF      = 3072
dtype   = float32
```

The test validates intermediate tensors and the final block output: input LayerNorm, Q/K/V projections, causal multi-head attention, output projection, residual path, second LayerNorm, feed-forward projection, GELU, second feed-forward projection, and final residual output.

All comparisons passed with an absolute tolerance of `5e-4`. In the reported run, the largest absolute error was approximately `4.77e-7` for the second LayerNorm and `7.45e-8` for the final output.

This is evidence of correctness for the tested configuration, not a proof for all shapes or inputs. The test uses deterministic inputs and weights, and the CPU reference shares the intended mathematical design. Randomized tests, more shape coverage, and independent checks remain useful. This validates one Transformer block, not a complete pretrained GPT-2 model.

### Tokenizer validation

The tokenizer wrapper uses `mlc-ai/tokenizers-cpp` and the GPT-2 tokenizer JSON file.

The test reports a vocabulary size of `50257` and verifies round-trip behavior for inputs including:

```text
Hello world!
Hello  world
It's a test.
Next line.
GPU optimization: 2× faster
```

The tests confirm that tokenization and decoding preserve the tested spaces, newlines, and Unicode characters.

### Token-embedding validation

The `Embedding<T>` layer maps token IDs to vectors stored in a device-resident embedding table. Its numerical test uses a small vocabulary of four tokens and an embedding dimension of three. The table is populated with known values, and the output is compared with the expected vectors.

This validates row selection and output layout for the tested case. GPT-2's pretrained embedding weights have not yet been loaded.

## Goals

The project is intended to build understanding of GPU programming and ML internals by implementing the components needed for a small language-model inference pipeline.

The development path is:

1. GPU memory management and tensor storage
2. Basic tensor operations and matrix multiplication
3. Neural-network layers and activations
4. Softmax and LayerNorm
5. Scaled dot-product and multi-head attention
6. Causal attention and online softmax
7. A reusable Transformer block
8. Token embeddings and learned positional embeddings
9. GPT-2 pretrained-weight loading and model assembly
10. Final LayerNorm and language-model output
11. Tokenization and autoregressive generation
12. Further kernel optimization

Inference is the initial focus. Automatic differentiation and training may be explored later.

## Implemented functionality

### Device memory and tensors

`DeviceBuffer<T>` provides RAII-style ownership of SYCL USM device allocations. It uses `sycl::malloc_device`, releases owned memory automatically, and prevents accidental shallow copies.

`Tensor<T>` provides a minimal multidimensional tensor abstraction with shape metadata, element count, device storage, and access to the underlying device pointer.

Example:

```cpp
tinyml::Tensor<float> X(queue, {32, 128});
```

This creates a tensor with shape `[32, 128]`. Tensor data resides in device memory; host/device transfers are performed explicitly through copy operations.

### Tensor operations

GPU operations are implemented in `ops.hpp`. The current set includes:

- Host-to-device and device-to-host copies
- Elementwise arithmetic and addition
- Fill and scalar multiplication
- Bias addition
- ReLU
- Exact GELU and tanh-approximate GELU
- Serial and parallel softmax
- Serial and parallel LayerNorm
- Affine parallel LayerNorm
- Naive matrix multiplication
- Tiled matrix multiplication
- Tiled matrix multiplication with a transposed second operand
- Scaled tiled multiplication with a transposed second operand
- Tiled online-softmax weighted-value product from precomputed attention scores
- Fused scaled dot-product attention using online softmax
- Multi-head attention
- Strided multi-head attention on `[B, L, d_model]` tensors
- Causal strided multi-head attention

Operations return `sycl::event` objects where applicable. Event dependencies express ordering between asynchronous GPU operations.

### Matrix multiplication

Several matrix-multiplication kernels are available, including naive and tiled implementations. The tiled implementations use SYCL local memory to reuse matrix data within work-groups.

The transposed operation computes

\[
C = AB^T
\]

for `A` with shape `[M, K]`, `B` with shape `[N, K]`, and `C` with shape `[M, N]`. It avoids creating a physical transposed copy of `B`.

A scaled variant computes

\[
C = \frac{AB^T}{\sqrt{K}},
\]

which is the score calculation used by scaled dot-product attention when `A = Q`, `B = K`, and `K` is the attention head dimension. Local-memory padding is used in the transposed implementation to help reduce unfavorable bank-access patterns.

These kernels are learning and optimization exercises, not replacements for tuned BLAS libraries.

### Linear layers

`Linear<T>` implements a fully connected layer:

\[
Y = XW + b
\]

where `X` has shape `[batch_size, in_features]`, `W` has shape `[in_features, out_features]`, `b` has shape `[out_features]`, and `Y` has shape `[batch_size, out_features]`.

The layer also supports Transformer-style 3D tensors:

```text
X [B, L, in_features]
        |
        v
      Linear
        |
        v
Y [B, L, out_features]
```

The contiguous `[B, L, in_features]` tensor is treated as a `[B*L, in_features]` matrix for multiplication. No physical reshape or data movement is required.

### Activation functions

ReLU and two GELU formulations are implemented as SYCL kernels.

ReLU:

\[
\operatorname{ReLU}(x) = \max(0, x)
\]

Exact GELU:

\[
\operatorname{GELU}(x) =
\frac{x}{2}
\left(1 + \operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
\]

Tanh approximation:

\[
\operatorname{GELU}(x) \approx
\frac{x}{2}
\left[
1+\tanh\left(\sqrt{\frac{2}{\pi}}(x+0.044715x^3)\right)
\right]
\]

The implementations have been compared against CPU references. Exact and approximate GELU have also been compared for numerical accuracy and execution time using SYCL event profiling.

### Softmax and LayerNorm

Both serial and parallel softmax implementations are available. The parallel version uses work-group reductions to normalize rows. It is used by the unfused attention path, while online-softmax state is used by the tiled and fused attention kernels.

Serial and parallel LayerNorm implementations are also available. The affine parallel version supports learned parameters:

\[
Y = \gamma \frac{X-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta
\]

It is used in the pre-LayerNorm Transformer block.

### Attention implementations

The project contains multiple attention implementations intended to make the algorithmic and memory trade-offs visible.

#### Unfused scaled dot-product attention

The operation is

\[
\operatorname{Attention}(Q,K,V) =
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V.
\]

The unfused path computes the score matrix with tiled `QK^T`, applies parallel softmax, and multiplies the weights by `V`. It is a useful reference implementation, but materializes the score and softmax-weight matrices.

#### Tiled online-softmax weighted-value product

`softmax_matmul_tiled` computes `O = softmax(S)V` from precomputed scores `S`. It streams over sequence tiles and maintains a running maximum, normalization factor, and output accumulator. When a new tile changes the running maximum, the accumulated state is rescaled to preserve numerical consistency.

This avoids materializing the complete softmax-weight matrix, but still requires the score matrix in global memory. It is an intermediate step toward fully fused attention.

#### Fused attention with online softmax

The fused online-softmax implementation processes blocks of `K` and `V` while maintaining the softmax state incrementally. For a new score tile with maximum \(m_{\text{tile}}\), the running maximum is updated as

\[
m_{\text{new}} = \max(m_{\text{old}}, m_{\text{tile}}).
\]

The denominator and output accumulator are rescaled consistently as new tiles are processed. This avoids storing full score and softmax-weight matrices.

The implementation is inspired by the blockwise online-softmax approach used by memory-efficient algorithms such as FlashAttention. It is an educational implementation, not a reproduction of an optimized FlashAttention kernel. In particular, the current implementation updates the running output accumulator through global device memory; a more optimized implementation could keep output fragments in registers or private storage.

#### Multi-head and strided attention

The project supports an explicit head-separated representation `[B, H, L, d_head]` and a strided representation that operates directly on projected tensors:

```text
Q [B, L, d_model]
K [B, L, d_model]
V [B, L, d_model]
```

The feature dimension is logically divided among heads:

\[
d_{\text{head}} = \frac{d_{\text{model}}}{H}.
\]

For head `h`, the feature offset is `h * d_head`. The strided kernel uses the full `d_model` row stride while accessing only the features belonging to the selected head. It therefore avoids a physical rearrangement to `[B, H, L, d_head]`.

The implementation launches attention independently for each batch/head pair and joins the resulting SYCL events into a completion event.

#### Causal multi-head attention

The causal kernel restricts each query at position `i` to keys and values at positions `j <= i`. Future positions are excluded from the softmax. It operates directly on `[B, L, d_model]` projected tensors and retains the strided head layout.

Causal attention has been tested at component level and integrated into the Transformer block. A useful invariant is that the first token can attend only to itself, so its attention output equals its value vector for each head.

#### Attention memory experiments

For square unfused attention with sequence length `N`, the score and softmax-weight matrices each have shape `[N, N]`. With FP32, those two matrices alone require `8*N*N` bytes.

Previous experiments on the development Intel integrated GPU ran the unfused implementation at `N = 32768`, where the two matrices together require approximately 8 GiB, and encountered a SYCL integer-range check at `N = 49152`. The fused implementation ran through `N = 65536` in the same experiment. These are observations from a particular experiment, not general hardware limits or performance claims.

### Transformer block

`TransformerBlock<T>` encapsulates the complete pre-LayerNorm block forward path. It owns the attention projections and feed-forward layers and exposes `forward()` to compose the operations through SYCL event dependencies.

The data path is:

```text
X [B, L, d_model]
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

The block has been validated against CPU reference calculations at the following dimensions:

```text
B       = 2
L       = 1024
d_model = 768
heads   = 12
d_head  = 64
d_ff    = 3072
```

The test compares intermediate results as well as the final output. In the reported run, all comparisons passed with absolute tolerance `5e-4`; the largest absolute error was approximately `4.77e-7` for the second LayerNorm and `7.45e-8` for the final output. The inputs and weights in this test are deterministic. Broader randomized and multi-shape testing remains part of strengthening validation.

This validates a single block at GPT-2 Small dimensions. It does not mean that the complete GPT-2 model is implemented.

### Tokenizer

`GPT2Tokenizer` is a C++ wrapper around `mlc-ai/tokenizers-cpp`. It loads a GPT-2 `tokenizer.json` file and provides methods to encode text into token IDs, decode token IDs into text, and query the vocabulary size.

Example:

```cpp
tinyml::GPT2Tokenizer tokenizer("models/gpt2/tokenizer.json");

const auto ids = tokenizer.encode("Hello world!");
const auto text = tokenizer.decode(ids);
```

The tokenizer test reports a vocabulary size of `50257` and verifies round-trip behavior for several text inputs, including repeated spaces, newlines, and Unicode characters.

The tokenizer is a component of the inference pipeline. It does not load model weights or run the GPT-2 neural network.

### Token embeddings

`Embedding<T>` implements a token embedding lookup using a device-resident weight table with shape `[vocab_size, embedding_dim]`.

For token IDs with shape `[B, L]`, the output has shape `[B, L, embedding_dim]`. Each token ID selects a row from the embedding table.

The initial numerical test uses a vocabulary of four tokens and an embedding dimension of three. It copies known token IDs and embedding weights to device memory, runs the lookup, copies the output back, and checks it against expected values.

This verifies the tested lookup and output layout. The GPT-2 embedding table has not yet been loaded from pretrained weights. Learned positional embeddings and their addition to token embeddings are the next implementation steps.

## Testing and benchmarks

Correctness tests are separated from performance experiments. The current test layout includes:

```text
tests/
├── test_core.cpp
├── test_nn.cpp
├── test_sdpa.cpp
├── test_attention.cpp
├── test_flash_attention.cpp
├── test_transformer_block.cpp
├── test_embedding.cpp
├── test_gpt2_tokenizer.cpp
├── benchmark_ops.cpp
└── benchmark_flash_attention.cpp
```

The intended responsibilities are:

- `test_core.cpp` — tensor operations and matrix multiplication
- `test_nn.cpp` — activations, softmax, LayerNorm, Linear, and feed-forward/MLP operations
- `test_sdpa.cpp` — scaled dot-product attention and fused online-softmax attention
- `test_attention.cpp` — multi-head, strided, causal, and attention integration tests
- `test_flash_attention.cpp` — CPU-reference comparisons for tiled attention operations, including online-softmax weighted-value products
- `test_transformer_block.cpp` — Transformer-block integration and numerical validation
- `test_embedding.cpp` — token embedding lookup with known weights and expected output
- `test_gpt2_tokenizer.cpp` — GPT-2 tokenizer loading and encode/decode round-trip tests
- `benchmark_ops.cpp` — performance experiments for operations such as GELU, tiled matrix multiplication, and attention
- `benchmark_flash_attention.cpp` — attention performance experiments across sequence lengths and implementation variants

The exact contents of test executables can evolve as the project develops. Keep correctness checks and performance benchmarks separate so performance experiments do not obscure validation results.

## GPT-2 Small target

GPT-2 Small is the first intended pretrained-model target. Its standard architecture is:

```text
vocabulary size        = 50257
context length         = 1024
d_model                = 768
attention heads        = 12
d_head                 = 64
Transformer blocks     = 12
feed-forward dimension = 3072
```

The computational operations for one GPT-2-style block are implemented, and the block forward pass has passed a CPU-reference comparison at these dimensions. The tokenizer and initial token embedding lookup are also implemented and tested independently.

The complete model still requires:

- Learned positional embeddings
- Addition of token and positional embeddings
- Loading pretrained GPT-2 weights
- Assembly of the 12 Transformer blocks
- Final LayerNorm
- The language-model head, including the correct weight tying behavior
- Integration of tokenization with model inference
- Autoregressive generation

The current milestone is a collection of validated components, not yet an end-to-end pretrained GPT-2 inference pipeline.

## Building

A SYCL-capable C++ compiler is required. Development currently uses Intel oneAPI `icpx`.

### Main project

Configure with tests enabled:

```bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=icpx \
  -DTINYML_BUILD_TESTS=ON
```

Build the main executable:

```bash
cmake --build build --target tinyml
```

Build the existing correctness tests:

```bash
cmake --build build --target test_core test_nn test_sdpa test_attention test_flash_attention test_transformer_block test_embedding
```

Run the tests:

```bash
./build/test_core
./build/test_nn
./build/test_sdpa
./build/test_attention
./build/test_flash_attention
./build/test_transformer_block
./build/test_embedding
```

Build the benchmark executables:

```bash
cmake --build build --target benchmark_ops benchmark_flash_attention
```

Run the benchmarks:

```bash
./build/benchmark_ops
./build/benchmark_flash_attention
```

These commands assume the corresponding targets are enabled in the main `CMakeLists.txt`. Check the current CMake configuration if a target name differs.

### GPT-2 tokenizer test

The tokenizer is built in a separate CMake configuration under `tests/`, where `tokenizers-cpp` is added as a dependency.

Configure and build:

```bash
cmake -S tests -B build/test_tokenizer
cmake --build build/test_tokenizer -j2
```

Run:

```bash
./build/test_tokenizer/test_gpt2_tokenizer
```

This configuration is separate from the main project build because the tokenizer test links against `tokenizers-cpp`.

The test expects the GPT-2 tokenizer JSON at:

```text
models/gpt2/tokenizer.json
```

Ensure this file exists at the expected path relative to the working directory.

### Compiler and target settings

The main project currently uses C++20 and Intel SYCL compilation flags, including `-fsycl` and `-fsycl-targets=spir64`. The precise compiler and target settings may need to change when using another SYCL implementation or device.

## Design principles

- C++20
- SYCL for GPU programming
- USM device memory
- Explicit `nd_range` kernels where appropriate
- Asynchronous operations linked through `sycl::event`
- Minimal abstractions
- GPU operations separated from neural-network layers
- Correctness before optimization
- CPU references used to validate GPU results
- Separate correctness tests and performance benchmarks

The purpose is to make the path from a neural-network operation to the underlying SYCL kernel understandable.

## Project structure

The following is a representative structure of the current project:

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
│       ├── linear.hpp
│       ├── embedding.hpp
│       ├── gpt2_tokenizer.hpp
│       ├── flash_attention.hpp
│       └── transformer_block.hpp
├── src/
│   └── main.cpp
├── models/
│   └── gpt2/
│       └── tokenizer.json
├── external/
│   └── tokenizers-cpp/
├── tests/
│   ├── test_core.cpp
│   ├── test_nn.cpp
│   ├── test_sdpa.cpp
│   ├── test_attention.cpp
│   ├── test_flash_attention.cpp
│   ├── test_transformer_block.cpp
│   ├── test_embedding.cpp
│   ├── test_gpt2_tokenizer.cpp
│   ├── benchmark_ops.cpp
│   └── benchmark_flash_attention.cpp
└── build/
```

This is a representative structure; adjust it to match the files present in the current checkout.

## Roadmap

### GPT-2 inference

- [x] Implement device memory management and basic tensor storage
- [x] Implement core tensor operations and matrix multiplication
- [x] Implement Linear layers and activation functions
- [x] Implement softmax and LayerNorm
- [x] Implement causal multi-head attention
- [x] Implement a reusable Transformer block
- [x] Validate one Transformer block at GPT-2 Small dimensions
- [x] Implement the GPT-2 tokenizer wrapper
- [x] Test token embedding lookup with known weights
- [ ] Implement learned positional embeddings
- [ ] Add token and positional embeddings
- [ ] Load pretrained GPT-2 weights
- [ ] Assemble the complete 12-block GPT-2 Small model
- [ ] Implement final LayerNorm
- [ ] Implement the language-model head and weight tying
- [ ] Run an end-to-end pretrained GPT-2 inference test
- [ ] Implement autoregressive generation

### Correctness and optimization

- [ ] Add randomized inputs and weights with multiple seeds to Transformer-block correctness tests
- [ ] Expand shape coverage across batch size, sequence length, embedding dimension, head count, and feed-forward width
- [ ] Profile and optimize bottlenecks after correctness is established
- [ ] Compute `QK^T` tiles on the fly and avoid materializing the score matrix
- [ ] Keep attention output fragments in registers or private storage where practical
- [ ] Tune work-group mapping and synchronization
- [ ] Explore a Triton version of the attention algorithm to compare its programming model with SYCL
- [ ] Explore portability to another SYCL implementation such as AdaptiveCpp
- [ ] Consider automatic differentiation and training as a longer-term extension

## Status summary

Implemented and tested components include USM device memory management, multidimensional tensors, host/device transfers, elementwise operations, naive and tiled matrix multiplication, linear layers, ReLU and GELU, serial and parallel softmax, affine parallel LayerNorm, unfused scaled dot-product attention, tiled online-softmax weighted-value products, fused online-softmax attention, multi-head and strided attention, causal attention, Q/K/V projections, attention output projection, residual connections, the Transformer feed-forward path, and a reusable `TransformerBlock<T>` forward pass.

The GPT-2 tokenizer wrapper and an initial token embedding lookup are also implemented and tested independently.

The current milestone is a numerically validated single Transformer block at GPT-2 Small dimensions, together with independently tested tokenization and token embedding components. The next major milestone is assembling these components into a complete pretrained GPT-2 Small inference pipeline, including learned positional embeddings, pretrained weights, all 12 Transformer blocks, the language-model head, and autoregressive generation.
