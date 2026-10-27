
/*
cmake -S . -B build -DTINYML_BUILD_TESTS=ON;cmake --build build --target test_embedding -j2;./build/test_embedding
*/
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "tinyml/embedding.hpp"

int main()
{
    try {
        sycl::queue queue{sycl::default_selector_v};
        tinyml::Embedding<float> embedding(queue, 4, 3);
        tinyml::Tensor<std::int32_t> token_ids(queue, {2, 2});
        tinyml::Tensor<float> output(queue, {2, 2, 3});

        const std::vector<std::int32_t> host_ids = {2, 0, 3, 1};
        const std::vector<float> host_weights = {0.1f, 0.2f, 0.3f, 1.1f, 1.2f, 1.3f, 2.1f, 2.2f, 2.3f, 3.1f, 3.2f, 3.3f};

        queue.memcpy(token_ids.data(), host_ids.data(), host_ids.size() * sizeof(std::int32_t));
        queue.memcpy(embedding.weights().data(), host_weights.data(), host_weights.size() * sizeof(float)).wait();

        embedding.forward(token_ids, output).wait_and_throw();

        std::vector<float> actual(output.size());
        queue.memcpy(actual.data(), output.data(), actual.size() * sizeof(float)).wait_and_throw();

        const std::vector<float> expected = {2.1f, 2.2f, 2.3f, 0.1f, 0.2f, 0.3f, 3.1f, 3.2f, 3.3f, 1.1f, 1.2f, 1.3f};

        for (std::size_t i = 0; i < actual.size(); ++i) {
            std::cout << actual[i] << (i + 1 == actual.size() ? '\n' : ' ');
            if (std::abs(actual[i] - expected[i]) > 1e-5f) {
                throw std::runtime_error("Embedding output mismatch at index " + std::to_string(i));
            }
        }

        std::cout << "Embedding test passed.\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Embedding test failed: " << e.what() << '\n';
        return 1;
    }
}
