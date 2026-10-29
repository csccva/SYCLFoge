/*
cmake -S . -B build -DCMAKE_CXX_COMPILER=icpx -DTINYML_BUILD_TESTS=ON -DTINYML_BUILD_GPT2_WEIGHTS_TEST=ON -DTINYML_BUILD_GPT2_BLOCK_TEST=ON -DTINYML_BUILD_GPT2_MODEL_TEST=ON;cmake --build build --target test_gpt2_model -j2;./build/test_gpt2_model
*/
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "tinyml/tensor.hpp"
#include "tinyml/ops.hpp"
#include "tinyml/gpt2_model.hpp"

int main() {
    try {
        sycl::queue queue{sycl::gpu_selector_v};

        constexpr std::size_t B = 1;
        constexpr std::size_t L = 4;
        constexpr std::size_t VOCAB_SIZE = tinyml::GPT2Model::VOCAB_SIZE;

        std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << '\n';
        std::cout << "Loading GPT-2 Small...\n";

        tinyml::GPT2Model model(queue, "models/gpt2");

        std::vector<std::int32_t> host_tokens{15496, 995, 0, 50256};
        tinyml::Tensor<std::int32_t> token_ids(queue, {B, L});
        tinyml::Tensor<float> logits(queue, {B, L, VOCAB_SIZE});

        ops::copy_to_device(queue, token_ids, host_tokens.data()).wait_and_throw();

        std::cout << "Running all 12 Transformer blocks...\n";
        auto e_logits = model.forward(token_ids, logits);
        e_logits.wait_and_throw();

        std::vector<float> host_logits(B * L * VOCAB_SIZE);
        ops::copy_to_host(queue, host_logits.data(), logits).wait_and_throw();

        bool all_finite = true;
        float min_logit = INFINITY;
        float max_logit = -INFINITY;

        for (float value : host_logits) {
            if (!std::isfinite(value)) {
                all_finite = false;
                break;
            }
            min_logit = std::min(min_logit, value);
            max_logit = std::max(max_logit, value);
        }

        std::cout << "Logits shape: [" << B << ", " << L << ", " << VOCAB_SIZE << "]\n";
        std::cout << "Total logits: " << host_logits.size() << '\n';

        if (!all_finite) {
            std::cerr << "GPT-2 MODEL TEST FAILED: non-finite logits detected\n";
            return 1;
        }

        const std::size_t last_position = (L - 1) * VOCAB_SIZE;
        const auto first = host_logits.begin() + static_cast<std::ptrdiff_t>(last_position);
        const auto last = first + static_cast<std::ptrdiff_t>(VOCAB_SIZE);
        const auto best = std::max_element(first, last);
        const std::size_t predicted_token = static_cast<std::size_t>(best - first);

        std::cout << "Minimum logit: " << min_logit << '\n';
        std::cout << "Maximum logit: " << max_logit << '\n';
        std::cout << "Next-token argmax ID: " << predicted_token << '\n';
        std::cout << "Next-token maximum logit: " << *best << '\n';
        std::cout << "GPT-2 FULL MODEL SMOKE TEST PASSED\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "GPT-2 model test error: " << e.what() << '\n';
        return 1;
    }
}
