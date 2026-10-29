# tinyml: Development and Validation Guide

This document explains how to build, run, test, benchmark, and debug `tinyml`.

It complements the other two project documents:

- `README.md` describes the project and its current capabilities.
- `docs/ARCHITECTURE_AND_TUTORIAL.md` explains SYCL, GPU programming, neural-network operations, Transformers, and GPT-2.
- This document explains the practical development workflow and records important validation results and debugging lessons.

The instructions assume a Linux environment, Intel oneAPI `icpx`, an available SYCL GPU device, and the project root as the working directory.

## 1. Development environment

The project uses C++20 and SYCL. The current development configuration uses Intel oneAPI `icpx` to compile for the `spir64` target.

The repository includes the `tokenizers-cpp` dependency under:

```text
external/tokenizers-cpp/
```

The GPT-2 inference pipeline also requires prepared model parameters and tokenizer files under:

```text
models/gpt2/
```

Python is used for model-parameter export, reference calculations, numerical comparisons, and an optional tokenizer wrapper. The native C++ inference executable does not require Python while generating text.

### Python dependencies

The Python tools use some or all of the following packages:

- NumPy
- PyTorch
- Transformers
- safetensors

Activate the project's virtual environment:

```bash
source .venv/bin/activate
```

Check the installed packages:

```bash
python -c "import numpy, torch, transformers, safetensors; print('Python dependencies OK')"
```

If PyTorch is absent and is needed for reference inference, a CPU-only installation can be used:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The reference model runs on the CPU, so it does not require a working PyTorch GPU backend.

A warning from Transformers that PyTorch is unavailable is not necessarily a problem when running only `tools/query_gpt2.py`. That script uses the Transformers tokenizer and delegates neural-network inference to the C++ executable. The numerical reference script, by contrast, requires PyTorch.

## 2. Configure and compile the project

Run these commands from the repository root.

### Configure the build

The following configuration enables the main test suite, the GPT-2 tests, text generation, and the logit diagnostic:

```bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=icpx \
  -DTINYML_BUILD_TESTS=ON \
  -DTINYML_BUILD_GPT2_WEIGHTS_TEST=ON \
  -DTINYML_BUILD_GPT2_BLOCK_TEST=ON \
  -DTINYML_BUILD_GPT2_MODEL_TEST=ON \
  -DTINYML_BUILD_GPT2_GENERATE=ON \
  -DTINYML_BUILD_GPT2_DUMP_LOGITS=ON
```

The project uses CMake options to control which targets are created. Reconfiguring an existing build directory updates its configuration but does not automatically rebuild every target.

If a target is missing, inspect the available targets:

```bash
cmake --build build --target help
```

### Build the main executable

```bash
cmake --build build --target tinyml -j2
```

### Build GPT-2 text generation

```bash
cmake --build build --target gpt2_generate -j2
```

### Build the numerical diagnostic

```bash
cmake --build build --target gpt2_dump_logits -j2
```

### Build the correctness tests

```bash
cmake --build build --target \
  test_core \
  test_nn \
  test_sdpa \
  test_attention \
  test_flash_attention \
  test_transformer_block \
  test_embedding \
  test_positional_embedding \
  test_gpt2_embedding \
  test_gpt2_embedding_validation \
  test_gpt2_weights \
  test_gpt2_block \
  test_gpt2_model \
  -j2
```

This command assumes that all those targets are enabled and defined in the current CMake configuration.

### Build the benchmarks

```bash
cmake --build build --target \
  benchmark_ops \
  benchmark_flash_attention \
  -j2
```

CMake may report that Doxygen is not installed. If the configuration and generation complete successfully, that warning does not by itself prevent compilation.

## 3. Run the correctness tests

Correctness tests check whether an operation or model produces the expected result. They are different from benchmarks, which measure execution time.

### Core tensor operations

```bash
./build/test_core
```

This target tests the core tensor functionality and numerical operations included in `tests/test_core.cpp`.

### Neural-network operations

```bash
./build/test_nn
```

This target tests neural-network operations such as activation functions, softmax, LayerNorm, and linear transformations.

It is particularly relevant to the LayerNorm numerical-stability issue described later in this document.

### Attention

```bash
./build/test_sdpa
./build/test_attention
./build/test_flash_attention
```

These tests cover scaled dot-product attention and the attention implementations represented by their corresponding test files.

### Transformer block

```bash
./build/test_transformer_block
```

The Transformer block test validates an integrated block against CPU reference calculations. The tested GPT-2 Small dimensions include a hidden dimension of 768, 12 heads, and a feed-forward dimension of 3072.

This test isolates one block. It does not, on its own, validate the complete pretrained model.

### Embeddings

```bash
./build/test_embedding
./build/test_positional_embedding
./build/test_gpt2_embedding
./build/test_gpt2_embedding_validation
```

These targets validate the embedding operations available in the current codebase.

### GPT-2 pretrained weights

```bash
./build/test_gpt2_weights
```

This target checks the loading of the prepared GPT-2 embedding weights and the associated embedding forward pass.

### GPT-2 Transformer block with pretrained weights

```bash
./build/test_gpt2_block
```

The GPT-2 block test loads a selected pretrained block and compares its output against a generated CPU reference.

The test supports selecting a layer index:

```bash
./build/test_gpt2_block 0
./build/test_gpt2_block 11
```

The corresponding exported parameters and reference input/output files must exist. The block export and reference-generation tools prepare these files.

For example:

```bash
python tools/export_gpt2_block.py --layer 0
python tools/reference_gpt2_block.py --layer 0 --sequence-length 4
./build/test_gpt2_block 0
```

Repeat the export and reference preparation with the desired layer index before testing another block.

### Full GPT-2 model

```bash
./build/test_gpt2_model
```

This is the full-model smoke test. It checks that the model can be loaded, run, and used to produce logits with the expected shape.

A successful smoke test is useful, but numerical comparison against a trusted reference provides stronger evidence of correctness.

### Interpreting test failures

If a test fails:

1. Read the first relevant error message.
2. Identify whether the failure is in configuration, compilation, file loading, kernel execution, or a numerical comparison.
3. Reproduce the issue with the smallest relevant test.
4. Fix and retest the affected component before relying on the complete model again.

Do not interpret a build failure as a numerical failure, or a poor generated sentence as proof of a kernel bug.

## 4. Run GPT-2 inference

The native C++ generator accepts a text prompt, uses the C++ tokenizer, runs inference through SYCL, and decodes the generated token IDs in C++.

Run it from the project root, because the model files are referenced using paths relative to that directory.

```bash
./build/gpt2_generate --max-new-tokens 20 "The capital of Finland is"
```

The output includes the selected SYCL device, model-loading messages, generated token IDs, and decoded text.

The generator currently uses greedy decoding. At each step, it selects the token with the largest logit. It recomputes the full input context for every new token and does not use a KV cache.

A repeated phrase does not automatically indicate an implementation error. GPT-2 Small is a base language model rather than an instruction-tuned chatbot, and greedy decoding can encourage repetition.

### Try another prompt

```bash
./build/gpt2_generate --max-new-tokens 20 "Hello"
```

```bash
./build/gpt2_generate --max-new-tokens 20 "Once upon a time"
```

```bash
./build/gpt2_generate --max-new-tokens 20 \
  "The quick brown fox jumps over the lazy dog."
```

The prompt and requested output must fit within GPT-2's maximum context length of 1024 tokens.

### Python tokenizer wrapper

The optional wrapper uses Hugging Face's GPT-2 tokenizer to create token IDs, then passes those IDs to the C++ executable:

```bash
source .venv/bin/activate

python tools/query_gpt2.py \
  --max-new-tokens 20 \
  "The capital of Finland is"
```

In this workflow:

1. Python tokenizes the prompt.
2. Python passes the integer token IDs to C++.
3. C++ executes the GPT-2 model using SYCL.
4. Python decodes the generated IDs for display.

The neural-network inference still runs in C++/SYCL.

The executable supports both direct C++ tokenization and the `--token-ids` mode used by the wrapper. Its output should indicate which input mode was selected.

## 5. Run performance benchmarks

The project contains separate benchmark executables for general operations and attention.

Build them if they have not already been built:

```bash
cmake --build build --target benchmark_ops benchmark_flash_attention -j2
```

Run the general operation benchmarks:

```bash
./build/benchmark_ops
```

Run the attention benchmarks:

```bash
./build/benchmark_flash_attention
```

The benchmarks explore the execution time of selected kernels and implementations. Depending on the benchmark, these include matrix multiplication, activation functions, and attention variants.

### How to interpret benchmark results

A benchmark measures performance for a particular implementation, input shape, device, and runtime environment. Results may change with:

- Matrix and sequence dimensions
- Tile size and work-group size
- Global-memory traffic
- Local-memory use
- Kernel launch overhead
- Synchronization
- Compiler options
- Device workload and thermal conditions

A kernel that wins for one input size may lose for another.

Record the device, relevant dimensions, implementation variant, and measurement method when comparing results. Avoid claiming that one kernel is universally faster from a single measurement.

### Correctness before performance

Keep correctness and performance testing separate.

When changing a kernel:

1. Run the relevant correctness test.
2. Confirm that its numerical result remains acceptable.
3. Run the relevant benchmark.
4. Compare the result with the previous implementation under the same conditions.

A faster kernel is not an improvement if it calculates the wrong result.

## 6. Numerical comparison against Hugging Face GPT-2

The full-model reference test compares all 50,257 vocabulary logits from the C++/SYCL implementation against logits from Hugging Face GPT-2 using the local checkpoint.

This is stronger than checking only the decoded text or the highest-scoring token.

### Files used by the comparison

The scripts use these files:

```text
build/reference_input_ids.txt
build/reference_logits.bin
build/sycl_logits.bin
```

The reference script writes the token IDs and reference logits. The SYCL diagnostic consumes those exact token IDs and writes the GPU model's logits. The comparison script reads both output vectors.

### Step 1: Generate reference logits

Activate the Python environment:

```bash
source .venv/bin/activate
```

Choose a prompt and run the reference model on the CPU:

```bash
python tools/reference_gpt2_logits.py "The capital of Finland is"
```

The script loads `models/gpt2/model.safetensors`, tokenizes the prompt using the local GPT-2 tokenizer, runs the reference model, and writes:

```text
build/reference_input_ids.txt
build/reference_logits.bin
```

The output should report the exact input token IDs, the logit range, and the highest-scoring token predictions.

### Step 2: Generate SYCL logits

```bash
./build/gpt2_dump_logits
```

The diagnostic reads `build/reference_input_ids.txt`, runs the C++/SYCL model, and writes:

```text
build/sycl_logits.bin
```

This ensures that both implementations process exactly the same token IDs.

### Step 3: Compare all logits

```bash
python tools/compare_gpt2_logits.py
```

The comparison reports:

- Maximum absolute error
- Mean absolute error
- Root-mean-square error (RMSE)
- Mean signed error
- Error after subtracting each vector's maximum logit
- Cosine similarity
- Pearson correlation
- Top-1 prediction agreement
- Top-10 overlap

For reference logits \(r_i\) and SYCL logits \(s_i\), the absolute error is:

\[
e_i = |s_i-r_i|.
\]

The maximum absolute error is:

\[
e_{\max} = \max_i |s_i-r_i|.
\]

The RMSE is:

\[
\operatorname{RMSE}
=
\sqrt{
\frac{1}{V}
\sum_{i=1}^{V}(s_i-r_i)^2
},
\]

where the vocabulary size is \(V=50257\).

These metrics quantify how closely the two numerical outputs agree.

### Important: run the comparison steps in order

Each reference run overwrites the previous reference files. Each SYCL diagnostic run overwrites the previous SYCL logits.

For every new prompt, run these three commands in order:

```bash
python tools/reference_gpt2_logits.py "Hello"
./build/gpt2_dump_logits
python tools/compare_gpt2_logits.py
```

Then repeat with another prompt.

Do not generate several reference outputs first and then run the SYCL diagnostic once. That would compare outputs from different prompts.

### Observed full-model validation results

The following results were observed when comparing the current SYCL implementation against the local Hugging Face checkpoint:

| Prompt | Token count | Maximum absolute error | RMSE | Top-1 agreement | Top-10 overlap |
|---|---:|---:|---:|---|---|
| `Hello` | 1 | 0.0003052 | 0.0002148 | Yes | 10/10 |
| `The capital of Finland is` | 5 | 0.0001907 | 0.0000452 | Yes | 10/10 |
| `Nice to meet you . My name is cris. What's your name?` | 15 | 0.0001984 | 0.0000434 | Yes | 10/10 |
| `The quick brown fox jumps over the lazy dog. This is a longer test of the model.` | 19 | 0.0003357 | 0.0000854 | Yes | 10/10 |

The maximum absolute errors were approximately \(1.9\times10^{-4}\) to \(3.4\times10^{-4}\). The top-10 prediction sets matched exactly for the tested prompts.

These results provide strong evidence that the full forward pass reproduces the reference model for these inputs. They do not prove correctness for every possible input, sequence length, batch size, or device configuration.

The reference and SYCL implementations use the same local checkpoint, so this test validates the computation against the loaded weights, not the quality of those weights or the language-generation behavior.

## 7. LayerNorm numerical-stability case study

This section records a debugging issue encountered during development. It illustrates how a mathematically correct formula can still require care when implemented using floating-point arithmetic.

### 7.1 What LayerNorm calculates

LayerNorm normalizes each row of a tensor.

For a row containing \(N\) elements \(x_j\), the mean is:

\[
\mu = \frac{1}{N}\sum_{j=0}^{N-1}x_j.
\]

The variance is:

\[
\sigma^2 = \frac{1}{N}
\sum_{j=0}^{N-1}(x_j-\mu)^2.
\]

The normalized output with learned scale and bias is:

\[
y_j =
\gamma_j
\frac{x_j-\mu}{\sqrt{\sigma^2+\epsilon}}
+\beta_j.
\]

The project has serial and parallel implementations of LayerNorm. Both must calculate sufficiently accurate row statistics before normalizing the input.

### 7.2 The failing case

A numerical test exposed an accuracy problem with a constant row containing 1000 elements, all equal to 1000:

```text
N = 1000
x[j] = 1000 for every j
group size = 256
```

The correct mean is exactly 1000, and the variance is zero.

However, computing the mean by directly accumulating the original values can lose precision in floating-point arithmetic. The parallel implementation also has to combine partial sums produced by different work-items.

This case was especially useful because the mathematical result is known in advance. It provides a simple test for whether the reduction calculates the mean accurately.

### 7.3 The correction: calculate a shifted mean

Instead of summing the original values, choose a reference value from the row:

\[
r = x_0.
\]

Then calculate the mean using centered values:

\[
\mu =
r + \frac{1}{N}\sum_{j=0}^{N-1}(x_j-r).
\]

This expression is mathematically equivalent to the ordinary mean because:

\[
\begin{aligned}
r + \frac{1}{N}\sum_j(x_j-r)
&= r + \frac{1}{N}\left(\sum_jx_j-Nr\right) \\
&= r + \frac{1}{N}\sum_jx_j-r \\
&= \frac{1}{N}\sum_jx_j.
\end{aligned}
\]

The numerical behavior is different because the reduction now accumulates values relative to the reference rather than accumulating the original values directly.

For the constant-row test, each centered value is zero:

\[
x_j-r = 1000-1000=0.
\]

The centered reduction therefore returns zero, and the computed mean becomes:

\[
\mu = r + 0 = 1000.
\]

This avoids accumulating the large common offset across the entire row.

### 7.4 Serial implementation

The serial LayerNorm implementation now follows this pattern:

```cpp
const T reference = input[i_start];

T centered_sum = 0;
for (std::size_t j = 0; j < N; ++j) {
    centered_sum += input[i_start + j] - reference;
}

const T mean =
    reference + centered_sum / static_cast<T>(N);
```

The implementation then calculates the variance relative to the resulting mean and applies the usual normalization.

The important change is the calculation of the mean. It does not change the intended LayerNorm equation.

### 7.5 Parallel implementation

The parallel implementation uses the same mathematical strategy but distributes the centered sum across work-items.

Each work-item accumulates a subset of the row:

```cpp
const T reference = input[row * N];

T row_sum = 0;
for (std::size_t j = local_j; j < N; j += group_size) {
    row_sum += input[row * N + j] - reference;
}
```

The work-items then reduce their partial sums using the existing work-group reduction. Once the partial sums have been combined, the mean is calculated as:

```cpp
const T mean =
    reference + local_A[0] / static_cast<T>(N);
```

Here, `local_A[0]` represents the reduced centered sum in the implementation.

The variance calculation and normalization follow the corrected mean.

The serial and parallel versions therefore use the same numerical idea, while preserving their different execution strategies.

### 7.6 How the correction was validated

After the change, the neural-network tests passed:

```bash
cmake --build build --target test_nn -j2
./build/test_nn
```

The observed results included:

```text
Parallel LayerNorm maximum error: approximately 3.58e-7
Serial-versus-parallel maximum difference: approximately 1.19e-6
```

These measurements refer to the tested cases, not a guaranteed error bound for every shape or input.

The constant-row test checks a particularly sensitive case, while comparisons between serial and parallel implementations provide another way to detect numerical inconsistencies.

### 7.7 General lesson

The LayerNorm issue demonstrates three important principles:

1. **Mathematical equivalence does not guarantee identical floating-point behavior.** Different evaluation strategies can accumulate different rounding errors.
2. **Simple inputs can expose numerical bugs.** A constant row gives a known mean and variance, making the test easy to interpret.
3. **A numerical fix must be validated.** The corrected formula was followed by serial/parallel comparisons and the existing neural-network tests.

The shift used here improves mean accuracy for the tested large-offset case. It is not a universal guarantee that every floating-point reduction is accurate under all conditions.

## 8. Troubleshooting

### CMake cannot find the compiler

Check that Intel oneAPI's compiler environment is active and that `icpx` is available:

```bash
which icpx
icpx --version
```

If the compiler is not found, initialize the oneAPI environment using the setup script provided by your installation before configuring CMake.

### A CMake target is missing

Inspect the available targets:

```bash
cmake --build build --target help
```

Then configure the relevant option to `ON` and re-run CMake. For example:

```bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=icpx \
  -DTINYML_BUILD_GPT2_GENERATE=ON
```

### No SYCL GPU is available

The project selects a GPU using:

```cpp
sycl::queue queue{sycl::gpu_selector_v};
```

If no compatible GPU is available, queue creation may fail. Verify that your SYCL runtime and device driver can see the intended GPU.

### Model files cannot be opened

The executable expects model and tokenizer paths relative to the repository root. Run commands from that directory.

Check that the expected files exist:

```bash
ls -lh models/gpt2/model.safetensors
ls -lh models/gpt2/tokenizer/tokenizer.json
ls -lh models/gpt2/token_embeddings.bin
ls -lh models/gpt2/position_embeddings.bin
```

The exported block directories and final LayerNorm parameter files are also required by the full model.

If exported files are missing, use the relevant parameter-export scripts before running inference.

### The Python wrapper produces a tokenizer warning

A message saying PyTorch is unavailable is not necessarily an error when running `tools/query_gpt2.py`, because that script uses a tokenizer and delegates inference to C++.

For numerical reference inference, PyTorch is required. Test its availability with:

```bash
python -c "import torch; print(torch.__version__)"
```

### Python and C++ report different token counts

Check which interface is being used.

For the native C++ prompt interface:

```bash
./build/gpt2_generate --max-new-tokens 20 "The capital of Finland is"
```

For the Python-tokenizer path, the wrapper must pass integer IDs using the `--token-ids` option. The C++ executable should then report that it is using token IDs supplied by the caller.

If the C++ program interprets the numeric token IDs as a text prompt, the resulting tokens will be wrong and the numerical comparison will be invalid.

### The numerical comparison reports unexpectedly large errors

Confirm that the reference and SYCL models use:

- The same checkpoint
- The same vocabulary and tokenizer
- The exact same input token IDs
- The same final sequence position
- The same output layout and FP32 interpretation

Run the reference script, SYCL diagnostic, and comparison in that order. Each script overwrites its output files, so stale files or a mismatched prompt can lead to a meaningless comparison.

If the top token matches but the errors are unexpectedly large, compare the full vectors and inspect the diagnostic's tensor shape and memory offset. A matching top token alone does not prove that the remaining logits are correct.

### Generated text repeats itself

Repetition alone is not evidence of a numerical bug.

The current generator uses greedy decoding. GPT-2 Small is also a base language model, not an instruction-tuned chatbot. First confirm numerical agreement against the reference. Only then experiment with decoding methods such as temperature sampling, top-k sampling, or repetition penalties.

## 9. Recommended development workflow

Use this sequence when adding or changing a component.

1. **Understand the intended mathematics.** Write down the equation and expected input/output shapes.
2. **Implement the smallest useful operation.** Avoid changing many components at once.
3. **Test simple cases first.** Use known values and cases where the expected output can be calculated directly.
4. **Compare serial and parallel implementations when appropriate.** This helps isolate reduction and indexing mistakes.
5. **Run the relevant component test.** For example, use `test_nn` for a LayerNorm change.
6. **Run the integrated test.** Verify that the component still works in a Transformer block or the full model.
7. **Compare the full-model logits when the change affects inference.** Use the reference scripts.
8. **Benchmark only after correctness passes.** Record the workload and the measurement.
9. **Document the result.** Explain what changed, why, and which tests support the conclusion.

This workflow keeps the project understandable and makes failures easier to locate.

## 10. Current validation boundary

The full GPT-2 forward pass has been compared with the reference model on multiple prompts and sequence lengths. The observed logit errors were small, and the top-10 predictions matched for each tested input.

The Transformer block has also passed separate CPU-reference checks at GPT-2 Small dimensions.

These are strong results for the tested configurations. They do not establish correctness for every possible input, every supported shape, or every device.

Continue to expand shape coverage, randomized tests, and model-level numerical comparisons as the implementation evolves.

The guiding principle is:

**Make the computation correct, demonstrate that with tests, and only then optimize how it executes.**
