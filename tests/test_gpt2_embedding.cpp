
/*
cmake -S . -B build -DTINYML_BUILD_TESTS=ON;cmake --build build --target test_gpt2_embedding -j2;./build/test_gpt2_embedding
*/
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include "tinyml/tensor.hpp"
#include "tinyml/gpt2_embedding.hpp"

int main() {
    sycl::queue queue{sycl::gpu_selector_v};

    constexpr std::size_t B = 2, L = 2, D = 3, V = 4;
    tinyml::GPT2Embedding<float> embedding(queue, V, L, D);

    tinyml::Tensor<std::int32_t> token_ids(queue, {B, L});
    tinyml::Tensor<float> token_vectors(queue, {B, L, D});
    tinyml::Tensor<float> output(queue, {B, L, D});

    const std::vector<std::int32_t> ids{2, 0, 3, 1};
    const std::vector<float> token_weights{
        0.1f, 0.2f, 0.3f,
        1.1f, 1.2f, 1.3f,
        2.1f, 2.2f, 2.3f,
        3.1f, 3.2f, 3.3f
    };
    const std::vector<float> position_weights{
        10.0f, 20.0f, 30.0f,
        40.0f, 50.0f, 60.0f
    };

    queue.memcpy(token_ids.data(), ids.data(), ids.size() * sizeof(std::int32_t));
    queue.memcpy(embedding.token_weights().data(), token_weights.data(), token_weights.size() * sizeof(float));
    queue.memcpy(embedding.position_weights().data(), position_weights.data(), position_weights.size() * sizeof(float)).wait();

    embedding.forward(token_ids, token_vectors, output).wait();

    std::vector<float> actual(B * L * D);
    queue.memcpy(actual.data(), output.data(), actual.size() * sizeof(float)).wait();

    const std::vector<float> expected{
        12.1f, 22.2f, 32.3f,
        40.1f, 50.2f, 60.3f,
        13.1f, 23.2f, 33.3f,
        41.1f, 51.2f, 61.3f
    };

    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(actual[i] - expected[i]) > 1e-4f) {
            std::cerr << "Mismatch at index " << i
                      << ": expected " << expected[i]
                      << ", got " << actual[i] << '\n';
            return 1;
        }
    }

    std::cout << "GPT2 embedding test passed.\n";
    return 0;
}
