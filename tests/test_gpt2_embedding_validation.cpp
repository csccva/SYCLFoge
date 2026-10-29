/*
cmake -S . -B build -DTINYML_BUILD_TESTS=ON;cmake --build build --target test_gpt2_embedding_validation -j2;./build/test_gpt2_embedding_validation
*/
#include <sycl/sycl.hpp>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include "tinyml/tensor.hpp"
#include "tinyml/gpt2_embedding.hpp"

template <typename F>
bool throws_invalid_argument(F&& f)
{
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

int main()
{
    sycl::queue queue{sycl::gpu_selector_v};
    tinyml::GPT2Embedding<float> embedding(queue, 8, 4, 3);

    tinyml::Tensor<std::int32_t> valid_ids(queue, {1, 2});
    tinyml::Tensor<float> valid_vectors(queue, {1, 2, 3});
    tinyml::Tensor<float> valid_output(queue, {1, 2, 3});

    tinyml::Tensor<std::int32_t> long_ids(queue, {1, 5});
    tinyml::Tensor<float> long_vectors(queue, {1, 5, 3});
    tinyml::Tensor<float> long_output(queue, {1, 5, 3});

    tinyml::Tensor<std::int32_t> zero_ids(queue, {1, 0});
    tinyml::Tensor<float> zero_vectors(queue, {1, 0, 3});
    tinyml::Tensor<float> zero_output(queue, {1, 0, 3});

    if (!throws_invalid_argument([&] { embedding.forward(long_ids, long_vectors, long_output); })) {
        std::cerr << "Failed: sequence longer than max_seq_len was accepted\n";
        return 1;
    }

    if (!throws_invalid_argument([&] { embedding.forward(zero_ids, zero_vectors, zero_output); })) {
        std::cerr << "Failed: zero-sized sequence was accepted\n";
        return 1;
    }

    if (!throws_invalid_argument([&] { embedding.forward(valid_ids, valid_vectors, valid_output, 0); })) {
        std::cerr << "Failed: group_size=0 was accepted\n";
        return 1;
    }

    std::cout << "GPT2 embedding validation tests passed.\n";
    return 0;
}
