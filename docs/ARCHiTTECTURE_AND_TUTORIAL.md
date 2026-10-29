# tinyml: Architecture and Beginner's Guide

This document explains the `tinyml` codebase from the ground up. It assumes no prior knowledge of SYCL, GPU programming, Transformers, or GPT-2.

The goal is not merely to describe what the program does. It is to explain how data moves through the system, how mathematical operations are implemented, how the components fit together, and how the implementation is checked for correctness.

## 1. What is tinyml?

`tinyml` is a small machine-learning framework written in C++20 and SYCL.

It implements the components needed to execute a neural network:

- Storage for numerical arrays
- GPU memory management
- Mathematical operations on arrays
- Neural-network layers
- Attention
- Transformer blocks
- A complete GPT-2 Small model
- Text generation

A conventional machine-learning framework hides much of this machinery. A program might ask a framework to calculate a matrix multiplication, and the framework chooses an implementation internally.

In `tinyml`, the relevant abstractions and kernels are explicit. This makes it possible to study the relationship between a mathematical operation and the code that executes it on a GPU.

The project has two separate objectives:

1. **Correctness:** Reproduce the intended neural-network computations.
2. **Performance:** Execute those computations efficiently on the GPU.

These objectives are related, but they are not the same. A program can produce correct output and still be slow. It can also run quickly while implementing the wrong computation. The project validates correctness before attempting more aggressive optimization.

## 2. Your first runs

This section shows how to compile the project, generate text using GPT-2, run a correctness test, and compare the SYCL model against a reference implementation.

The commands assume a Linux environment, Intel oneAPI `icpx`, an available SYCL GPU device, and that the current working directory is the root of the `tinyml` repository.

### 2.1 Compile the GPT-2 generator

Configure CMake:

```bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=icpx \
  -DTINYML_BUILD_GPT2_GENERATE=ON
```

Build the executable:

```bash
cmake --build build --target gpt2_generate -j2
```

The resulting executable is:

```text
build/gpt2_generate
```

CMake only needs to be reconfigured when you change build options or build configuration. You normally rebuild after modifying source code.

### 2.2 Run a simple prompt

Run:

```bash
./build/gpt2_generate --max-new-tokens 10 "Hello"
```

The program tokenizes the prompt in C++, loads the pretrained GPT-2 Small parameters, runs inference through SYCL on your GPU, and generates up to ten new tokens.

Try a longer prompt:

```bash
./build/gpt2_generate --max-new-tokens 20 "The capital of Finland is"
```

Or:

```bash
./build/gpt2_generate --max-new-tokens 20 "Once upon a time"
```

The program prints the selected device, generated token IDs, and decoded text.

The model may produce repetitive or grammatically poor output. GPT-2 Small is a base language model rather than an instruction-tuned chatbot, and the current generator uses greedy decoding. Poor text quality does not automatically indicate an implementation error.

### 2.3 Understand what happens during a run

The execution path is:

```text
Text prompt
    |
    v
C++ GPT-2 tokenizer
    |
    v
Token IDs
    |
    v
Copy input IDs to GPU
    |
    v
GPT-2 inference through SYCL
    |
    v
Copy final-position logits to CPU
    |
    v
Select highest-scoring token
    |
    v
Append token to context
    |
    v
Repeat inference
    |
    v
Decode generated token IDs in C++
    |
    v
Generated text
```

The tokenizer is implemented in C++ using `tokenizers-cpp`. The neural network's forward pass runs through your C++/SYCL implementation. Python is not required for this native generation path.

### 2.4 Run a correctness test

For example, build and run the neural-network tests:

```bash
cmake --build build --target test_nn -j2
./build/test_nn
```

To test the integrated Transformer block:

```bash
cmake --build build --target test_transformer_block -j2
./build/test_transformer_block
```

These tests focus on individual components. The full-model numerical comparison described below checks whether the complete GPT-2 forward pass reproduces the reference model's logits.

Other available tests and their purposes are discussed in `docs/DEVELOPMENT_AND_VALIDATION.md`.

### 2.5 Compare the model against a reference

The numerical comparison requires the Python environment and its reference dependencies.

Activate the environment:

```bash
source .venv/bin/activate
```

Generate reference logits for a prompt:

```bash
python tools/reference_gpt2_logits.py "Hello"
```

Run the SYCL model on the same token IDs:

```bash
./build/gpt2_dump_logits
```

Compare the results:

```bash
python tools/compare_gpt2_logits.py
```

The comparison checks all 50,257 vocabulary logits, not just the highest-scoring token.

Run these commands in order for every new prompt. The scripts overwrite their previous output files.

The reference test requires PyTorch, Transformers, NumPy, and safetensors. If PyTorch is missing, install it in the active virtual environment. A CPU-only PyTorch package is sufficient because the reference model runs on the CPU.

For the complete compilation, test, benchmark, and validation workflow, see `docs/DEVELOPMENT_AND_VALIDATION.md`.

## 3. What are a CPU, GPU, compiler, and SYCL?

### 3.1 CPU

The CPU is the general-purpose processor that executes ordinary program instructions.

It manages program control flow, file operations, memory allocation, command-line arguments, and other tasks. In this project, the CPU also chooses the next token during generation by finding the largest value in the output logits.

The CPU is good at executing relatively complex sequences of instructions and coordinating the program.

### 3.2 GPU

A GPU is a processor designed to execute many similar operations in parallel.

Matrix multiplication is a good example. A large matrix may contain millions of output elements. Computing each output element involves a similar calculation, so many elements can be calculated concurrently.

Neural networks contain many matrix multiplications and elementwise operations. GPUs are therefore useful for executing neural-network computations.

This project uses an Intel integrated GPU. Integrated means that the GPU is part of the same processor package as the CPU, rather than being a separate graphics card with its own dedicated memory hardware.

### 3.3 Compiler

A compiler translates source code into a form that can execute on a target machine.

For this project, `icpx` is the Intel C++ compiler with SYCL support. It compiles the C++ source code, including the SYCL device kernels.

The C++ source remains ordinary source code in files such as:

```text
src/gpt2_generate.cpp
src/gpt2_dump_logits.cpp
include/tinyml/ops.hpp
include/tinyml/transformer_block.hpp
include/tinyml/gpt2_model.hpp
```

### 3.4 SYCL

SYCL is a C++ programming model for programming heterogeneous computing devices, including CPUs and GPUs.

Heterogeneous means that a program can use more than one type of processor. Here, the CPU runs program control code and the Intel GPU performs the main neural-network computations.

SYCL provides several important abstractions:

- A **device** represents the processor on which work can execute.
- A **queue** is used to submit work to a device.
- A **kernel** is a function executed by the device.
- A **buffer or USM allocation** provides memory that can be accessed according to the chosen memory model.
- An **event** represents submitted work and allows the program to express dependencies.

A simplified example of creating a GPU queue is:

```cpp
sycl::queue queue{sycl::gpu_selector_v};
```

The queue selects a GPU device and provides an interface for submitting work to it.

The actual program does more than this: it allocates memory, submits operations, waits for some operations to finish, and transfers selected results back to the CPU.

### 3.5 What is a SYCL kernel?

A kernel is a function that executes on the device rather than as an ordinary CPU function.

For example, suppose we want to add two arrays:

\[
C_i = A_i + B_i.
\]

The addition for each index \(i\) is independent. A GPU kernel can assign different indices to different work-items, allowing many additions to execute concurrently.

The `tinyml` kernels use SYCL work-items and work-groups to implement operations such as elementwise addition, matrix multiplication, normalization, and attention.

The general principle is:

1. Determine what calculation must be performed.
2. Determine which parts of the calculation are independent.
3. Assign independent work to GPU work-items.
4. Coordinate work-items when they must share data or combine partial results.

Parallel reductions, such as summing a row of values, require more coordination than elementwise addition.

## 4. Understanding memory in this project

### 4.1 Host and device

In SYCL terminology, the host is the CPU-side program, and the device is the selected execution device.

The CPU and GPU do not necessarily access memory in the same way. The program must use a supported memory model and explicit operations to make data available where it is needed.

`tinyml` uses USM device allocations for tensor data.

### 4.2 USM

USM means Unified Shared Memory. It is a SYCL memory model that provides pointer-based memory access.

USM includes different allocation types. This project primarily uses **device allocations**, created with an operation such as:

```cpp
sycl::malloc_device<T>(count, queue);
```

A device allocation is intended for device access. It should not be treated as ordinary CPU-accessible memory.

To copy data from CPU memory to device memory, the program uses an explicit transfer operation. The reverse is required to retrieve results for CPU-side inspection.

For example, the model's input token IDs are copied to the device before the forward pass. The final-position logits are copied back to the CPU before greedy decoding.

### 4.3 `DeviceBuffer<T>`

`DeviceBuffer<T>` manages the lifetime of a device allocation.

Its main responsibility is ownership:

- Allocate the device memory.
- Retain the pointer and allocation size.
- Release the allocation when the owner is destroyed.
- Prevent unsafe copying of an object that owns the same allocation.

This is related to RAII in C++. RAII means that resource ownership is tied to an object's lifetime. When the object is destroyed, its resource is released.

Without an ownership abstraction, every caller would need to coordinate allocation and deallocation manually. That makes mistakes such as leaks and double-free operations more likely.

### 4.4 `Tensor<T>`

A tensor is a multidimensional array of values.

The template parameter `T` is the element type. For example:

```cpp
tinyml::Tensor<float> X(queue, {32, 128});
```

creates a tensor containing 32 × 128 floating-point values.

Its shape is:

```text
[32, 128]
```

The number of elements is:

\[
32 \times 128 = 4096.
\]

A tensor's shape describes the logical arrangement of its elements. The underlying storage is still a contiguous region of memory in this implementation.

A tensor with shape `[B, L, D]` can be interpreted as:

- `B`: batch size
- `L`: sequence length
- `D`: number of features per sequence position

For example, GPT-2 hidden states commonly have shape:

```text
[1, 5, 768]
```

This represents a batch containing one sequence of five tokens, with one 768-dimensional vector per token.

The tensor abstraction does not automatically make an operation mathematically correct. The kernel that accesses the tensor must use the shape and indexing consistently.

### 4.5 Why explicit copies matter

Consider the generation loop:

```text
CPU token IDs
     |
     v
Copy IDs to device
     |
     v
GPU model inference
     |
     v
Copy final-position logits to CPU
     |
     v
CPU selects the next token
```

The model's arithmetic occurs on the GPU. The token selection currently occurs on the CPU.

This is a deliberate and simple division of work. More advanced implementations might keep more of the generation state on the device, but that is a separate optimization problem.

## 5. Asynchronous execution and events

GPU operations are often submitted asynchronously. The CPU can enqueue work without waiting for every individual operation to finish immediately.

A SYCL event represents submitted work and can be used to express dependencies.

For example:

```cpp
auto event = ops::copy_to_device(queue, tensor, host_data);
event.wait_and_throw();
```

The copy is submitted to the queue, and the program waits for it to finish. `wait_and_throw()` also allows asynchronous errors associated with the work to be reported.

In a sequence of dependent calculations, the program must ensure that a consumer does not read a result before its producer has finished writing it.

For example:

```text
Copy input
   |
   v
Calculate Q, K, V
   |
   v
Calculate attention
   |
   v
Project attention output
```

The exact event dependencies are determined by the implementation of the relevant operations and layer methods.

Waiting after every operation is simple to reason about, but can reduce opportunities for overlap. A more advanced implementation may preserve explicit event dependencies without waiting on the CPU at every step.

This project uses events to make the ordering of GPU operations visible.

## 6. The mathematical building blocks

Before studying GPT-2, it helps to understand the numerical operations that make it work.

### 6.1 Elementwise operations

An elementwise operation applies a calculation independently to each element.

For example:

\[
C_i = A_i + B_i.
\]

Each output element depends only on the corresponding input elements. That makes elementwise operations straightforward to parallelize.

Other examples include scaling:

\[
Y_i = \alpha X_i
\]

and applying an activation function:

\[
Y_i = \operatorname{ReLU}(X_i).
\]

### 6.2 Matrix multiplication

A matrix multiplication computes:

\[
C = AB.
\]

For matrices \(A\) and \(B\), the element at row \(i\) and column \(j\) is:

\[
C_{ij} = \sum_k A_{ik}B_{kj}.
\]

Each output element is a dot product of one row of \(A\) and one column of \(B\).

Matrix multiplication is central to neural networks because a linear layer applies a learned transformation to the input features.

`tinyml` implements both a simple matrix multiplication kernel and tiled variants. The tiled variants aim to reuse input data within a work-group instead of repeatedly fetching every operand from global memory.

### 6.3 Linear layers

A linear layer applies a learned transformation:

\[
Y = XW + b.
\]

Here:

- \(X\) contains the input features.
- \(W\) contains the learned weights.
- \(b\) contains the bias.
- \(Y\) contains the output features.

In GPT-2, linear layers calculate the query, key, and value projections, the attention output projection, and the feed-forward transformations.

The `Linear<T>` class composes the corresponding matrix multiplication and bias addition operations.

### 6.4 Activation functions

A neural network often applies a nonlinear activation after a linear transformation.

GPT-2 uses GELU in its feed-forward network.

One common expression for GELU is:

\[
\operatorname{GELU}(x) =
\frac{x}{2}
\left(1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right).
\]

GPT-2 uses the tanh approximation:

\[
\operatorname{GELU}(x) \approx
\frac{x}{2}
\left[
1+\tanh\left(
\sqrt{\frac{2}{\pi}}(x+0.044715x^3)
\right)
\right].
\]

The project provides both exact and approximate implementations for experimentation. The GPT-2 feed-forward computation uses the formulation appropriate to the checkpoint and reference model.

### 6.5 Softmax

Softmax converts a row of scores into normalized positive values:

\[
p_i = \frac{e^{s_i}}{\sum_j e^{s_j}}.
\]

The result sums to one, aside from floating-point rounding.

In attention, the scores indicate how strongly a query should attend to different positions. Softmax converts the scores into weights used to combine the value vectors.

In numerically stable implementations, a value such as the row maximum is subtracted before exponentiation:

\[
p_i =
\frac{e^{s_i-m}}
{\sum_j e^{s_j-m}},
\qquad
m = \max_j s_j.
\]

Subtracting the same scalar from every score does not change the mathematical softmax result, but it reduces the risk of overflow when exponentiating large values.

### 6.6 LayerNorm

Layer normalization normalizes the features of each input row.

For one row \(x\) containing \(D\) features, its mean is:

\[
\mu = \frac{1}{D}\sum_{j=1}^{D}x_j.
\]

Its variance is:

\[
\sigma^2 = \frac{1}{D}\sum_{j=1}^{D}(x_j-\mu)^2.
\]

Normalization with learned scale and bias gives:

\[
y_j =
\gamma_j
\frac{x_j-\mu}{\sqrt{\sigma^2+\epsilon}}
+\beta_j.
\]

Here, \(\gamma\) and \(\beta\) are learned parameters, while \(\epsilon\) is a small constant used for numerical stability.

GPT-2 uses LayerNorm both within each Transformer block and after the final block.

### 6.7 LayerNorm debugging: constant-valued inputs

During development, the LayerNorm implementation was tested with a row containing 1000 values, each equal to 1000. The parallel version used a work-group size of 256.

This case is valuable because the correct mathematical answer is known exactly. It helps isolate problems in the mean, variance, or normalization calculation.

#### What should LayerNorm produce?

Consider a row containing \(N\) identical values:

\[
x_j=c,\qquad j=0,\ldots,N-1.
\]

For the tested input:

```text
Number of elements: 1000
Input value:        1000
Work-group size:    256
```

The mean is:

\[
\mu=\frac{1}{N}\sum_{j=0}^{N-1}x_j=c.
\]

Because every element equals the mean, each deviation is zero:

\[
x_j-\mu=0.
\]

The variance is consequently:

\[
\sigma^2=
\frac{1}{N}\sum_{j=0}^{N-1}(x_j-\mu)^2=0.
\]

Substituting these values into the LayerNorm equation gives:

\[
y_j =
\gamma_j\frac{c-c}{\sqrt{0+\epsilon}}+\beta_j
=\beta_j.
\]

**The expected result is the learned bias vector \(\beta\).** If the bias is zero, every output element should be zero.

This invariant does not depend on the row length or the constant's value, provided the values are represented identically and the calculations are performed correctly.

#### Why an inaccurate mean matters

Suppose the computed mean is slightly wrong:

\[
\widehat{\mu}=c+\delta.
\]

The calculated deviation is then:

\[
x_j-\widehat{\mu}=-\delta
\]

instead of zero.

The variance calculation may also be affected, and the resulting normalized output can differ from the expected bias vector.

This makes the constant-row case a useful test for detecting problems in the reduction used to calculate the mean, the variance calculation, or the normalization itself.

The original implementation failed this case. The failure established that the tested computation was incorrect, but did not by itself prove the exact underlying cause. Floating-point rounding, incorrect indexing, incomplete reductions, or synchronization problems can all produce unexpected results. In particular, summing 1000 identical values of `1000.0f` is exactly representable in FP32, so the failure should not automatically be attributed to ordinary summation rounding alone.

#### The correction: calculate a shifted mean

The conventional mean calculation is:

\[
\mu=\frac{1}{N}\sum_{j=0}^{N-1}x_j.
\]

The corrected implementation selects the first input value as a reference:

\[
r=x_0.
\]

It then calculates:

\[
\boxed{
\mu =
r+\frac{1}{N}\sum_{j=0}^{N-1}(x_j-r)
}
\]

This formula is mathematically equivalent to the ordinary mean:

\[
\begin{aligned}
\mu
&=r+\frac{1}{N}\sum_j(x_j-r)\\
&=r+\frac{1}{N}\left(\sum_jx_j-Nr\right)\\
&=r+\frac{1}{N}\sum_jx_j-r\\
&=\frac{1}{N}\sum_jx_j.
\end{aligned}
\]

The definition of LayerNorm has not changed. Only the numerical evaluation of the mean has changed.

For a constant-valued row, \(r=c\), so:

\[
x_j-r=c-c=0.
\]

The centered sum is therefore zero:

\[
\sum_j(x_j-r)=0.
\]

The calculated mean becomes:

\[
\mu=r+\frac{0}{N}=c.
\]

For rows whose values are close to a common reference, subtracting the reference can also reduce the magnitude of the values being accumulated, improving the numerical behavior of the reduction.

#### Serial implementation

The serial LayerNorm calculation uses this pattern:

```cpp
const T reference = input[i_start];

T centered_sum = 0;
for (std::size_t j = 0; j < N; ++j) {
    centered_sum += input[i_start + j] - reference;
}

const T mean =
    reference + centered_sum / static_cast<T>(N);
```

Here, `i_start` identifies the beginning of the row and `N` is the number of features.

The implementation then calculates the variance relative to the computed mean and applies the usual LayerNorm equation.

#### Parallel implementation

The parallel implementation uses the same mathematical strategy but distributes the centered sum across work-items.

Each work-item accumulates a subset of the row:

```cpp
const T reference = input[row * N];

T row_sum = 0;
for (std::size_t j = local_j; j < N; j += group_size) {
    row_sum += input[row * N + j] - reference;
}
```

Here:

- `row` identifies the row being normalized.
- `N` is the number of features in that row.
- `local_j` identifies the first element assigned to the work-item.
- `group_size` is the stride between elements assigned to the same work-item.

The work-items combine their partial sums through the existing work-group reduction. The mean is then calculated from the reduced centered sum:

```cpp
const T mean =
    reference + local_A[0] / static_cast<T>(N);
```

In this implementation, `local_A[0]` contains the reduced centered sum.

The variance calculation and normalization then proceed using the computed mean. The serial and parallel implementations thus use the same mathematical strategy while preserving their respective execution methods.

#### How the fix was validated

After the correction, the neural-network tests passed:

```bash
cmake --build build --target test_nn -j2
./build/test_nn
```

The reported results included:

```text
Parallel LayerNorm maximum error: approximately 3.58e-7
Serial-versus-parallel maximum difference: approximately 1.19e-6
```

These values describe the tested cases, not a guaranteed error bound for all input shapes.

The constant-row test checks a known mathematical invariant. Comparing serial and parallel implementations checks whether different execution strategies produce consistent results. The wider neural-network test suite checks that the change does not break other operations.

#### Lessons from this debugging case

1. **Test mathematical invariants.** Constant-valued rows have a known mean, zero variance, and predictable output.
2. **Separate the mathematical formula from its implementation.** Equivalent formulas can behave differently in floating-point computation.
3. **Use controlled inputs.** A simple test is easier to diagnose than an unexpected output from an entire Transformer.
4. **Validate a correction.** Run component tests and compare against independent calculations where possible.
5. **Do not overstate the diagnosis.** A numerical fix may correct an observed failure without proving the precise cause of every original failure.

The general debugging approach is to derive the expected result, construct a test that exposes a violation, change the smallest relevant calculation, and rerun the tests.

## 7. What is a Transformer?

A Transformer is a neural-network architecture that processes sequences using attention and feed-forward layers.

For a language model, the sequence consists of tokens. Tokens may represent whole words, word fragments, punctuation, or other text units.

The central idea is that a token's representation should depend on relevant information from earlier tokens. Attention supplies a mechanism for combining information from different sequence positions.

GPT-2 uses a decoder-only Transformer architecture. It processes the input sequence and predicts the next token at each position. Causal masking prevents the representation at a position from using future tokens.

The complete GPT-2 model contains 12 Transformer blocks stacked one after another.

## 8. Tokens and embeddings

### 8.1 Text is not directly processed by the neural network

The model does not receive the original text as a string. It receives integer token IDs.

For example, the tokenizer converts:

```text
Hello
```

into the GPT-2 token ID:

```text
15496
```

The exact tokenization depends on the tokenizer and the input text. A sentence is generally converted into several token IDs.

The vocabulary contains 50,257 token IDs, numbered from zero through 50,256.

### 8.2 The tokenizer

`GPT2Tokenizer` is a C++ wrapper around `mlc-ai/tokenizers-cpp`.

It loads:

```text
models/gpt2/tokenizer/tokenizer.json
```

Its main operations are:

```cpp
tokenizer.encode(text);
tokenizer.decode(token_ids);
tokenizer.vocab_size();
```

`encode()` converts a string into a sequence of token IDs. `decode()` converts token IDs back into text.

The tokenizer is not the neural network. It prepares model inputs and converts generated IDs back into a human-readable form.

### 8.3 Token embeddings

Each token ID selects a learned vector from the token-embedding matrix.

The token-embedding matrix has shape:

```text
[50257, 768]
```

There is one row per vocabulary token and 768 features per row.

If a token has ID \(t\), its vector is:

\[
e_t = E[t].
\]

Here, \(E\) is the embedding matrix.

The token embedding layer performs a lookup, not a matrix multiplication. The integer ID identifies which row to select.

### 8.4 Positional embeddings

The Transformer also needs information about the position of each token.

GPT-2 uses learned positional embeddings. Its positional embedding matrix has shape:

```text
[1024, 768]
```

A position \(p\) selects:

\[
p_e = P[p].
\]

The token and position vectors are added:

\[
x_p = E[t_p] + P[p].
\]

The resulting vector has 768 features.

The full embedding tensor therefore has shape:

```text
[B, L, 768]
```

where `B` is the batch size and `L` is the input sequence length.

## 9. A GPT-2 Transformer block

A GPT-2 block transforms a sequence of 768-dimensional hidden vectors into another sequence of the same shape.

For GPT-2 Small, its main components are:

- LayerNorm
- Query, key, and value projections
- Causal multi-head attention
- Attention output projection
- A residual connection
- A second LayerNorm
- A feed-forward network
- A second residual connection

Its main dimensions are:

```text
Hidden dimension:       768
Attention heads:         12
Features per head:       64
Feed-forward dimension: 3072
```

The input and output shapes are both:

```text
[B, L, 768]
```

This stable shape makes it possible to stack the blocks.

### 9.1 Pre-LayerNorm

GPT-2 uses a pre-LayerNorm arrangement: normalization occurs before the attention operation and before the feed-forward network.

The first operation is:

\[
X_n = \operatorname{LayerNorm}_1(X).
\]

The normalized tensor is used to calculate the attention projections.

### 9.2 Queries, keys, and values

The normalized input produces three tensors:

\[
Q = X_n W_Q + b_Q
\]

\[
K = X_n W_K + b_K
\]

\[
V = X_n W_V + b_V.
\]

Each has shape:

```text
[B, L, 768]
```

Conceptually:

- A **query** represents what a position is looking for.
- A **key** represents what a position can be matched against.
- A **value** represents information that can be collected from a position.

These descriptions are an intuition for the mathematical roles of the projections. The tensors are simply learned numerical representations.

### 9.3 Splitting the hidden features into heads

GPT-2 has 12 attention heads, each with 64 features:

\[
12 \times 64 = 768.
\]

Conceptually, each projection can be interpreted as:

```text
[B, L, 768]
        |
        v
[B, 12, L, 64]
```

An implementation does not have to physically rearrange the data to make this logical division.

The `tinyml` strided attention implementation accesses head features directly from the original projected tensors, using the appropriate row stride and head offset.

### 9.4 Scaled dot-product attention

For each head, the score between a query position and a key position is:

\[
S = \frac{QK^T}{\sqrt{d_k}},
\]

where \(d_k=64\).

Dividing by \(\sqrt{64}=8\) keeps the score scale more manageable.

The scores are converted into weights using softmax:

\[
A = \operatorname{softmax}(S).
\]

The attention output is then:

\[
O = AV.
\]

For each query position, this computes a weighted combination of the value vectors at the positions it is permitted to attend to.

### 9.5 Causal masking

GPT-2 predicts the next token using only the current token and preceding context.

When processing a sequence, position \(i\) must not use information from position \(j\) when \(j>i\).

Causal attention therefore allows:

\[
j \leq i.
\]

Future positions are excluded from the softmax. In implementations that use a score mask, their scores are set to a value that contributes zero probability after softmax.

Without this restriction, the model's representation of a token could use future tokens during training or evaluation, violating the autoregressive prediction setup.

### 9.6 Combining the heads

After the heads compute their outputs, their features are combined into the 768-dimensional hidden representation.

The attention output projection applies:

\[
A_o = OW_O + b_O.
\]

This learns how information from the different heads should be combined.

### 9.7 The first residual connection

The attention output is added to the original block input:

\[
X_1 = X + A_o.
\]

This is a residual connection.

The important property is that the original input can pass through the block in addition to being transformed. Residual connections help make deep networks easier to optimize and preserve information across layers.

### 9.8 The feed-forward network

The second part of the block starts with another LayerNorm:

\[
X_n' = \operatorname{LayerNorm}_2(X_1).
\]

The feed-forward network expands the feature dimension from 768 to 3072, applies GELU, and projects the result back to 768:

\[
F_1 = X_n' W_1 + b_1
\]

\[
F_g = \operatorname{GELU}(F_1)
\]

\[
F_2 = F_g W_2 + b_2.
\]

The shapes are:

```text
[B, L, 768]
      |
      v
[B, L, 3072]
      |
      v
    GELU
      |
      v
[B, L, 768]
```

The final block output is:

\[
Y = X_1 + F_2.
\]

The two residual paths and two LayerNorm operations are important parts of the GPT-2 block architecture.

### 9.9 The C++ implementation

`include/tinyml/transformer_block.hpp` defines `TransformerBlock<T>`.

It composes the operations needed for the block, including:

- LayerNorm
- Linear projections
- Causal attention
- Attention output projection
- Residual additions
- Feed-forward projections
- GELU

The block owns its learned parameters and exposes a `forward()` method that accepts the input and intermediate tensors.

The full GPT-2 model creates 12 blocks and runs them in sequence.

## 10. Attention memory strategies in tinyml

Attention is mathematically straightforward but can consume substantial memory if implemented by storing every intermediate matrix.

### 10.1 Unfused attention

The basic implementation calculates:

\[
S = \frac{QK^T}{\sqrt{d_k}}
\]

then:

\[
A = \operatorname{softmax}(S)
\]

and finally:

\[
O = AV.
\]

It materializes the score matrix and softmax-weight matrix.

For a sequence of length \(L\), each matrix contains \(L^2\) values per attention head and batch item.

With FP32 values, the two matrices together require approximately:

\[
8L^2 \text{ bytes}
\]

per batch/head pair, excluding other tensors and overhead.

This quadratic memory growth makes the unfused method expensive at long sequence lengths.

### 10.2 Tiled execution

A tiled kernel processes a portion of a larger matrix at a time.

The advantages include:

- Reusing data within a work-group
- Reducing repeated global-memory accesses
- Controlling how much intermediate data is handled at once

The trade-offs depend on tile size, work-group mapping, synchronization, and device capabilities.

### 10.3 Online softmax

Online softmax avoids requiring the entire score matrix to be stored before calculating the final weighted output.

The kernel processes scores in tiles and maintains running statistics:

- A running maximum
- A normalization factor
- A weighted-value accumulator

When a new tile changes the maximum, the accumulated state is rescaled so that it remains consistent with the new normalization.

The fused `tinyml` implementation follows this approach. It avoids storing the full softmax-weight matrix, although its current implementation still has global-memory traffic associated with the running output accumulator.

This approach is inspired by memory-efficient attention algorithms such as FlashAttention. It is an educational implementation rather than a claim of equivalent optimization or performance.

## 11. How the complete GPT-2 model is assembled

`include/tinyml/gpt2_model.hpp` defines the model-level forward pass.

GPT-2 Small contains 12 Transformer blocks. Every block uses its own pretrained weights.

The model's computation proceeds as follows.

### Step 1: Input token IDs

The input tensor has shape:

```text
[B, L]
```

For example, a five-token prompt with batch size one has shape:

```text
[1, 5]
```

The elements are integer token IDs.

### Step 2: Token and positional embeddings

The model looks up a 768-dimensional token vector and a 768-dimensional positional vector for every input position.

It adds the vectors to produce:

```text
[B, L, 768]
```

### Step 3: Twelve Transformer blocks

The hidden tensor passes through the 12 blocks in order:

```text
Embedding output
      |
      v
 Transformer block 0
      |
      v
 Transformer block 1
      |
      v
       ...
      |
      v
 Transformer block 11
```

Each block preserves the `[B, L, 768]` shape.

Although the tensor shape stays the same, the contents change at every block.

### Step 4: Final LayerNorm

After the last block, the model applies its final learned LayerNorm.

This normalizes the hidden representation before vocabulary prediction.

### Step 5: Vocabulary projection

Each 768-dimensional hidden vector is projected into a vector containing 50,257 scores:

```text
[B, L, 768]
      |
      v
[B, L, 50257]
```

These scores are called **logits**.

The GPT-2 implementation uses the token-embedding matrix for this output projection, following the checkpoint's weight-tying design.

### Step 6: Return logits

The model returns a logit vector for every input position.

For generation, the important vector is the one associated with the final input position. That vector represents the model's scores for possible next tokens.

The model does not generate text by itself. It produces numerical scores; a separate generation loop turns those scores into token IDs.

## 12. What are logits?

A logit is a numerical score produced before converting scores into probabilities.

For one output position, GPT-2 produces 50,257 logits, one for each vocabulary token.

A larger logit means the model assigns a higher relative score to that token, with the model's other logits held fixed.

Logits can be negative. Their absolute sign does not tell us whether the model is correct.

For example:

```text
Token A: -94.7
Token B: -95.2
Token C: -100.0
```

Token A has the largest logit among these three candidates.

For greedy decoding, the model selects:

\[
t_{\text{next}} = \operatorname{argmax}_i(\text{logit}_i).
\]

This means selecting the index with the largest score.

Logits can also be converted to probabilities using softmax, but the generation loop in this project currently selects the largest logit directly.

## 13. Autoregressive text generation

Autoregressive generation means generating one token at a time, using the tokens already available as the context for the next prediction.

Suppose the prompt is:

```text
The capital of Finland is
```

The tokenizer converts it into token IDs. The model processes the complete input and predicts the next token.

The selected token is appended to the context. The model then runs again to predict another token.

The process is:

```text
Initial prompt
      |
      v
Tokenize
      |
      v
Run GPT-2
      |
      v
Select next token
      |
      v
Append token to context
      |
      v
Run GPT-2 again
      |
      v
Select next token
      |
      v
     ...
```

This continues until the maximum number of new tokens is reached or the end-of-text token is generated.

### 13.1 Greedy decoding

The current implementation uses greedy decoding.

At each step, it selects the token whose logit is the largest:

\[
t_{\text{next}} = \operatorname{argmax}_i z_i.
\]

This is simple and deterministic: the same input and numerical result produce the same next token.

However, greedy decoding can produce repetitive text. It makes each choice independently according to the largest current score and does not explicitly discourage repeating a previous phrase.

### 13.2 Why output can repeat

Repetition can result from several factors:

- GPT-2 Small is a base next-token prediction model, not an instruction-tuned chatbot.
- Greedy decoding repeatedly selects the highest-scoring next token.
- A model can assign high scores to continuations that repeat previously generated text.
- The model's learned weights and the prompt affect which continuations are likely.

Repetitive text is therefore not sufficient evidence of a numerical bug.

In this project, the reference comparison provides more useful evidence: it checks whether the implementation computes the same logits as the reference model.

### 13.3 The current generation loop recomputes the context

The current implementation submits the entire current token sequence to GPT-2 at every generation step.

For each new token, it creates input and output tensors, copies token IDs to the GPU, runs the full forward pass, and copies the final-position logits back to the CPU.

This is conceptually straightforward, but it repeats work for tokens that were already processed.

An optimization called a **KV cache** stores the previously calculated attention keys and values. That can avoid recomputing all earlier positions at every step. It is not implemented in the current version.

## 14. Following the source code

This section describes where to look when studying the code. File details can evolve, so always check the current source.

### 14.1 `include/tinyml/device_buffer.hpp`

Start here to understand how the project owns device allocations.

Questions to consider:

- Where is memory allocated?
- Where is the pointer stored?
- How is the size represented?
- When is memory released?
- Why are copying and ownership important?

Understanding this file is useful because almost every GPU computation depends on correct memory lifetime management.

### 14.2 `include/tinyml/tensor.hpp`

Next, examine the tensor abstraction.

Look for how the class stores its shape and how it calculates the total number of elements.

Then inspect how it accesses the underlying device storage.

The main concept to understand is that shape is metadata describing the logical tensor. The data is stored separately in device memory.

### 14.3 `include/tinyml/ops.hpp`

This file contains many of the low-level numerical operations.

Start with a simple operation such as fill or addition before studying matrix multiplication.

For each operation, identify:

1. The input shapes.
2. The output shape.
3. The mathematical equation.
4. How the SYCL kernel maps work-items to data.
5. Which memory locations each work-item reads and writes.
6. How the operation signals completion.

For matrix multiplication, follow how each output element is calculated and how the tiled version reuses data.

For LayerNorm, trace the reduction used to calculate the row mean and variance. Section 6.7 explains the constant-valued-row test and shifted-mean correction.

For attention, identify where scores are calculated, where softmax is applied, and how the resulting weights are multiplied by values.

### 14.4 `include/tinyml/linear.hpp`

Study how a matrix multiplication and a bias addition become a reusable neural-network layer.

Follow the shapes:

```text
Input:   [B, L, in_features]
Weights: [in_features, out_features]
Bias:    [out_features]
Output:  [B, L, out_features]
```

Understand why the input can be treated as a `[B*L, in_features]` matrix without physically rearranging the tensor data.

### 14.5 `include/tinyml/gpt2_embedding.hpp`

This file assembles token and positional embeddings.

Trace how token IDs index the token-embedding table. Then follow how the position for each sequence element selects a positional vector.

Check where the two vectors are added and what shape the resulting hidden tensor has.

For GPT-2 Small, the last dimension is 768.

### 14.6 `include/tinyml/transformer_block.hpp`

This is the best place to understand how several operations combine into a Transformer.

Follow the `forward()` method in this order:

1. First LayerNorm.
2. Q/K/V projections.
3. Causal multi-head attention.
4. Attention output projection.
5. First residual addition.
6. Second LayerNorm.
7. First feed-forward projection.
8. GELU.
9. Second feed-forward projection.
10. Second residual addition.

At each stage, write down the expected tensor shape.

The hidden tensors should have shape `[B, L, 768]` except for the expanded feed-forward activation, which has shape `[B, L, 3072]`.

The input and output of the complete block have the same shape.

### 14.7 `include/tinyml/gpt2_model.hpp`

This file connects the block implementation to the full pretrained model.

Follow how it:

- Loads the pretrained embedding and normalization parameters.
- Constructs all 12 Transformer blocks.
- Applies token and positional embeddings.
- Passes the hidden tensor through the blocks.
- Applies final LayerNorm.
- Calculates vocabulary logits.

Pay particular attention to the last projection. A hidden vector of 768 features becomes a vector of 50,257 vocabulary logits.

### 14.8 `include/tinyml/gpt2_tokenizer.hpp`

This is the C++ interface to `tokenizers-cpp`.

It loads `tokenizer.json` and provides the encoding and decoding operations.

The tokenizer does not perform model inference. It produces the input token IDs that the model requires and converts the generated IDs back into text.

### 14.9 `src/gpt2_generate.cpp`

This is the end-to-end program for text generation.

Read it as a sequence of operations:

1. Parse the command-line arguments.
2. Tokenize the prompt.
3. Validate the number of tokens.
4. Create a SYCL GPU queue.
5. Load the GPT-2 model.
6. Create input and output tensors for the current context.
7. Copy token IDs to device memory.
8. Call `GPT2Model::forward()`.
9. Copy the final-position logits to the CPU.
10. Select the highest-scoring token.
11. Append that token and repeat.
12. Decode and print the result.

The generator provides the program control flow. The numerical model computation happens in `GPT2Model::forward()` and the operations it invokes.

### 14.10 `src/gpt2_dump_logits.cpp`

This diagnostic program is deliberately simpler than the text generator.

It reads token IDs, runs the model once, extracts the final-position logits, and writes them to a binary file.

It does not need to generate text. This makes it useful for numerical testing because it exposes the raw numerical output of the model.

## 15. How the pretrained weights are loaded

The project uses a local checkpoint:

```text
models/gpt2/model.safetensors
```

The checkpoint contains learned parameters, not executable code.

The export scripts convert relevant checkpoint tensors into binary files expected by the C++ model loader.

Examples include:

```text
models/gpt2/token_embeddings.bin
models/gpt2/position_embeddings.bin
models/gpt2/final_ln_gamma.bin
models/gpt2/final_ln_beta.bin
models/gpt2/block0/
...
models/gpt2/block11/
```

Each block directory contains the parameters needed by the corresponding Transformer block, including LayerNorm parameters, attention projection weights and biases, and feed-forward weights and biases.

The model loader reads these files and assigns the parameters to the corresponding C++ model components.

This arrangement separates two concerns:

- **Checkpoint preparation:** Loading the external checkpoint and exporting parameters into the format expected by the C++ program.
- **Runtime inference:** Loading the exported parameter files and executing the neural network in C++ and SYCL.

Python is used by the export and diagnostic scripts. The native C++ generation path does not need Python to execute the network.

## 16. Numerical validation against Hugging Face

A neural network that produces text is not necessarily computing the correct function. To validate the implementation, we compare its numerical output with a trusted reference.

The project uses Hugging Face GPT-2 with the local checkpoint as the reference.

### 16.1 Generate the reference

The script:

```text
tools/reference_gpt2_logits.py
```

loads the local checkpoint and the GPT-2 tokenizer. It tokenizes a prompt, runs the reference model on the CPU, and saves the logits from the final input position.

The relevant files are:

```text
build/reference_input_ids.txt
build/reference_logits.bin
```

The reference vector has 50,257 values.

### 16.2 Generate the SYCL output

The diagnostic executable:

```text
build/gpt2_dump_logits
```

reads the same input token IDs and runs the C++/SYCL model.

It saves the corresponding final-position logits to:

```text
build/sycl_logits.bin
```

Because both runs consume the same token IDs and the same local checkpoint, we can compare their numerical outputs directly.

### 16.3 Compare the vectors

The script:

```text
tools/compare_gpt2_logits.py
```

calculates measures such as maximum absolute error, mean absolute error, RMSE, centered-logit error, and correlation.

For reference logits \(r_i\) and SYCL logits \(s_i\), the absolute error at vocabulary index \(i\) is:

\[
e_i = |s_i-r_i|.
\]

Maximum absolute error is:

\[
e_{\max} = \max_i |s_i-r_i|.
\]

RMSE is:

\[
\operatorname{RMSE}
=
\sqrt{
\frac{1}{V}
\sum_{i=1}^{V}(s_i-r_i)^2
},
\]

where \(V=50257\).

These measurements describe how closely the numerical vectors agree.

The comparison also checks whether both implementations select the same highest-scoring token and whether their top-10 prediction sets agree.

### 16.4 Why raw logit values can be negative

Logits are unnormalized scores. They do not need to be positive.

The important point is whether the SYCL model reproduces the reference logits and their relative ordering, not whether the scores happen to be negative.

In the reported tests, both models produce nearly identical logit vectors. Maximum absolute errors are around a few times \(10^{-4}\), and the top-10 predictions match for all tested inputs.

This is strong evidence that the full forward pass is numerically correct for those inputs. It does not guarantee correctness for every possible sequence or shape, nor does it measure language-generation quality.

### 16.5 Repeatability

The reference comparison can be repeated for different prompts:

```bash
python tools/reference_gpt2_logits.py "Hello"
./build/gpt2_dump_logits
python tools/compare_gpt2_logits.py
```

Then repeat using another prompt.

Run the steps in order. The reference script overwrites the reference files, the SYCL diagnostic overwrites the SYCL output, and the comparison reads the two most recently generated files.

## 17. Why correct logits do not guarantee good text

Numerical correctness and text quality are different questions.

The current implementation reproduces the reference GPT-2 logits closely on the tested inputs. This establishes that the model computation is behaving as expected for those cases.

It does not establish that GPT-2 Small will produce useful or coherent responses to every prompt.

GPT-2 Small is a base language model trained to predict text continuations. It was not specifically designed to behave like a modern instruction-following chatbot.

The current decoding method is also greedy. At every step, it selects the single highest-logit token. This can produce repeated phrases even when the numerical implementation is correct.

Possible future experiments include temperature sampling, top-k or top-p sampling, repetition penalties, and a KV cache. Each changes a different part of the generation pipeline and should be tested separately.

The model weights and the GPU computation should not be changed merely to address a decoding behavior that may be caused by the generation strategy.

## 18. Performance considerations

The project uses SYCL to execute the main neural-network computations on the GPU. That does not automatically mean every operation is optimal.

Several factors influence performance:

- Kernel launch overhead
- Global-memory bandwidth
- Local-memory usage
- Work-group size
- Data reuse
- Synchronization
- Number of intermediate tensors
- CPU/GPU data transfers
- Repeated computation during generation

The unfused attention implementation materializes the score and probability matrices. The online-softmax implementation avoids storing the complete probability matrix, but the current implementation still performs global-memory updates of its running output state.

Generation also recomputes the entire context at every step rather than caching earlier keys and values.

These are opportunities for future optimization, not evidence that the computation is incorrect.

A useful sequence is:

1. Validate the mathematics.
2. Measure execution time and memory behavior.
3. Identify the dominant cost.
4. Change one implementation detail.
5. Repeat the correctness tests.
6. Benchmark again.

This prevents performance changes from silently breaking the numerical result.

## 19. Suggested learning order

The fastest way to understand this codebase is not to read every file from beginning to end. Study the data flow and mathematical operations in an order that builds on previous concepts.

### Stage 1: Memory and tensors

Read:

```text
include/tinyml/device_buffer.hpp
include/tinyml/tensor.hpp
```

Be able to explain device memory, tensor shape, element count, and host/device transfers.

### Stage 2: Simple operations

Read the implementations of fill, copy, elementwise addition, and scalar multiplication in `include/tinyml/ops.hpp`.

Understand how a SYCL kernel maps a work-item to one or more tensor elements.

### Stage 3: Matrix multiplication

Study the naive matrix multiplication before the tiled version.

Be able to derive:

\[
C_{ij} = \sum_k A_{ik}B_{kj}.
\]

Then identify which work-item calculates which output element and how the tiled version reuses data.

### Stage 4: Neural-network layers

Read:

```text
include/tinyml/linear.hpp
include/tinyml/embedding.hpp
```

Understand the difference between a linear transformation and an embedding lookup.

### Stage 5: Normalization and attention

Read the softmax, LayerNorm, and attention implementations in `ops.hpp` and their related source files.

Be able to explain why attention requires Q, K, and V; how the score matrix is formed; why the scale is used; and how causal masking works.

Use Section 6.7 to study the constant-valued LayerNorm test and the shifted-mean correction. Try to derive the corrected mean formula yourself before looking at the implementation.

### Stage 6: One Transformer block

Read:

```text
include/tinyml/transformer_block.hpp
```

Trace each intermediate tensor and record its shape.

Do not proceed until you can explain the two residual connections, the two LayerNorm operations, the attention sublayer, and the feed-forward sublayer.

### Stage 7: Complete GPT-2

Read:

```text
include/tinyml/gpt2_embedding.hpp
include/tinyml/gpt2_layer_norm.hpp
include/tinyml/gpt2_block_weights.hpp
include/tinyml/gpt2_model.hpp
```

Understand how the pretrained parameters are loaded, how the 12 blocks are assembled, and how the model converts hidden vectors into vocabulary logits.

### Stage 8: Generation

Read:

```text
include/tinyml/gpt2_tokenizer.hpp
src/gpt2_generate.cpp
```

Trace the complete path from a text prompt to token IDs, through SYCL inference, and back to decoded text.

### Stage 9: Correctness and performance

Run the reference logit comparison. Then study the benchmark programs and investigate how changing the implementation affects performance without changing its mathematical result.

## 20. Scope and limitations

The current project is an inference-oriented educational framework.

It does not currently implement:

- Automatic differentiation
- Gradient computation
- Backpropagation
- Optimizers
- Training or fine-tuning
- A KV cache for autoregressive generation
- Advanced sampling controls
- A general-purpose model-loading system for arbitrary architectures

The full GPT-2 forward pass has been numerically compared against a trusted reference on several input sequences. This is a meaningful milestone, but continued testing is needed across more inputs, sequence lengths, and device configurations.

The goal is not to reproduce every feature of a large machine-learning framework. It is to make the core path from a neural-network equation to a GPU kernel understandable.

## 21. Final perspective

There are three distinct things to understand in this project:

1. **The model's mathematics:** What GPT-2 calculates.
2. **The implementation:** How the C++ classes and SYCL kernels calculate it.
3. **The execution strategy:** How work, memory, events, and data transfers are organized to run on the GPU.

A correct implementation must agree with the mathematical model. A performant implementation must also use the hardware effectively.

The current `tinyml` implementation has a working pretrained GPT-2 Small inference path and has shown very close agreement with Hugging Face logits for the tested inputs.

That gives the project a sound foundation for studying the code, improving performance, and experimenting with alternative kernels and generation strategies.