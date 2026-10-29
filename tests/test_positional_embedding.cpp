/*
cmake -S . -B build -DTINYML_BUILD_TESTS=ON;cmake --build build --target test_positional_embedding -j2;./build/test_positional_embedding*
*/
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "tinyml/tensor.hpp"
#include "tinyml/positional_embedding.hpp"

int main() {
    sycl::queue queue{sycl::gpu_selector_v};

    constexpr std::size_t B = 2, L = 2, D = 3;
    tinyml::Tensor<float> input(queue, {B, L, D});
    tinyml::Tensor<float> output(queue, {B, L, D});
    tinyml::PositionalEmbedding<float> embedding(queue, 4, D);

    const std::vector<float> input_values{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
    const std::vector<float> position_values{1, 2, 3, 4, 5, 6};

    queue.memcpy(input.data(), input_values.data(), input_values.size() * sizeof(float));
    queue.memcpy(embedding.weights().data(), position_values.data(), position_values.size() * sizeof(float)).wait();

    embedding.forward(input, output).wait();

    std::vector<float> actual(B * L * D);
    queue.memcpy(actual.data(), output.data(), actual.size() * sizeof(float)).wait();

    const std::vector<float> expected{11, 22, 33, 44, 55, 66};

    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(actual[i] - expected[i]) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": expected " << expected[i] << ", got " << actual[i] << '\n';
            return 1;
        }
    }

    std::cout << "Positional embedding test passed.\n";
    return 0;
}
