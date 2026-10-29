/*
cmake -S . -B build -DTINYML_BUILD_TESTS=ON -DTINYML_BUILD_GPT2_WEIGHTS_TEST=ON;cmake --build build --target test_gpt2_weights -j2;./build/test_gpt2_weights
*/

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tinyml/tensor.hpp"
#include "tinyml/gpt2_embedding.hpp"
#include "tinyml/gpt2_weights.hpp"

float read_reference_value(const std::string& path, std::size_t index) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open reference file: " + path);
    file.seekg(static_cast<std::streamoff>(index * sizeof(float)));
    float value = 0.0f;
    if (!file.read(reinterpret_cast<char*>(&value), sizeof(value))) throw std::runtime_error("Cannot read reference value: " + path);
    return value;
}

void check_device_value(sycl::queue& queue, const float* device_data, const std::string& path, std::size_t index) {
    float actual = 0.0f;
    queue.memcpy(&actual, device_data + index, sizeof(float)).wait();
    const float expected = read_reference_value(path, index);
    if (actual != expected) throw std::runtime_error("Weight mismatch in " + path + " at index " + std::to_string(index));
    std::cout << path << " index " << index << ": " << actual << " OK\n";
}

int main() {
    try {
        sycl::queue queue{sycl::default_selector_v};
        const std::string token_path = "models/gpt2/token_embeddings.bin";
        const std::string position_path = "models/gpt2/position_embeddings.bin";
        constexpr std::size_t vocab_size = 50257;
        constexpr std::size_t max_seq_len = 1024;
        constexpr std::size_t D = 768;

        tinyml::GPT2Embedding<float> embedding(queue, vocab_size, max_seq_len, D);
        tinyml::load_gpt2_weights(queue, embedding, token_path, position_path);

        const std::size_t token_count = vocab_size * D;
        const std::size_t position_count = max_seq_len * D;

        check_device_value(queue, embedding.token_weights().data(), token_path, 0);
        check_device_value(queue, embedding.token_weights().data(), token_path, 12345);
        check_device_value(queue, embedding.token_weights().data(), token_path, token_count - 1);
        check_device_value(queue, embedding.position_weights().data(), position_path, 0);
        check_device_value(queue, embedding.position_weights().data(), position_path, 12345);
        check_device_value(queue, embedding.position_weights().data(), position_path, position_count - 1);

        // Small test sequence. These are valid GPT-2 vocabulary IDs.
        constexpr std::size_t B = 1;
        constexpr std::size_t L = 3;
        tinyml::Tensor<std::int32_t> token_ids(queue, {B, L});
        tinyml::Tensor<float> token_vectors(queue, {B, L, D});
        tinyml::Tensor<float> output(queue, {B, L, D});

        const std::vector<std::int32_t> ids = {15496, 995, 0}; // "Hello world!"
        queue.memcpy(token_ids.data(), ids.data(), ids.size() * sizeof(std::int32_t)).wait();

        embedding.forward(token_ids, token_vectors, output).wait();

        std::vector<float> actual_output(B * L * D);
        queue.memcpy(actual_output.data(), output.data(), actual_output.size() * sizeof(float)).wait();

        constexpr float tolerance = 1e-6f;
        for (std::size_t pos = 0; pos < L; ++pos) {
            for (std::size_t d = 0; d < D; ++d) {
                const std::size_t output_index = pos * D + d;
                const std::size_t token_index = static_cast<std::size_t>(ids[pos]) * D + d;
                const std::size_t position_index = pos * D + d;
                const float expected = read_reference_value(token_path, token_index) + read_reference_value(position_path, position_index);
                const float actual = actual_output[output_index];
                if (std::fabs(actual - expected) > tolerance) {
                    throw std::runtime_error("Embedding output mismatch at position " + std::to_string(pos) + ", feature " + std::to_string(d) + ": actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected));
                }
            }
        }

        std::cout << "GPT-2 weight loading tests passed.\n";
        std::cout << "GPT-2 embedding forward-pass test passed for " << L << " tokens and " << D << " features.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "GPT-2 weight loading test failed: " << e.what() << '\n';
        return 1;
    }
}
