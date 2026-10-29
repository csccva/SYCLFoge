#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tinyml/tensor.hpp"
#include "tinyml/ops.hpp"
#include "tinyml/gpt2_model.hpp"

int main(int argc, char* argv[])
{
    try {
        const std::string input_path =
            argc > 1 ? argv[1] : "build/reference_input_ids.txt";
        const std::string output_path =
            argc > 2 ? argv[2] : "build/sycl_logits.bin";

        // Read the exact token IDs used by the reference model.
        std::ifstream input(input_path);
        if (!input) {
            throw std::runtime_error(
                "Cannot open input IDs: " + input_path);
        }

        std::vector<std::int32_t> tokens;
        long long id;

        while (input >> id) {
            if (id < 0 ||
                id >= tinyml::GPT2Model::VOCAB_SIZE) {
                throw std::runtime_error("Invalid input token ID");
            }
            tokens.push_back(static_cast<std::int32_t>(id));
        }

        if (tokens.empty() ||
            tokens.size() > tinyml::GPT2Model::MAX_SEQ_LEN) {
            throw std::runtime_error(
                "Input must contain between 1 and 1024 token IDs");
        }

        std::cerr << "Input token count: " << tokens.size() << '\n';

        // Initialize the SYCL device and load the model.
        sycl::queue queue{sycl::gpu_selector_v};

        std::cerr << "SYCL device: "
                  << queue.get_device().get_info<
                         sycl::info::device::name>()
                  << '\n';

        tinyml::GPT2Model model(queue, "models/gpt2");

        const std::size_t L = tokens.size();
        const std::size_t V = tinyml::GPT2Model::VOCAB_SIZE;

        tinyml::Tensor<std::int32_t> token_ids(queue, {1, L});
        tinyml::Tensor<float> logits(queue, {1, L, V});

        auto e_input = ops::copy_to_device(
            queue, token_ids, tokens.data());
        e_input.wait_and_throw();

        auto e_logits = model.forward(token_ids, logits);
        e_logits.wait_and_throw();

        // Export logits for the final input position.
        std::vector<float> last_logits(V);
        const std::size_t offset = (L - 1) * V;

        queue.memcpy(
            last_logits.data(),
            logits.data() + offset,
            V * sizeof(float)
        ).wait_and_throw();

        // Ensure the diagnostic output is finite.
        for (const float value : last_logits) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Non-finite logit encountered");
            }
        }

        std::ofstream output(
            output_path, std::ios::binary | std::ios::trunc);

        if (!output) {
            throw std::runtime_error(
                "Cannot create output file: " + output_path);
        }

        output.write(
            reinterpret_cast<const char*>(last_logits.data()),
            static_cast<std::streamsize>(
                last_logits.size() * sizeof(float)));

        if (!output) {
            throw std::runtime_error("Failed to write logits");
        }

        // Print the ten highest predictions for a quick sanity check.
        std::vector<std::size_t> indices(V);
        for (std::size_t i = 0; i < V; ++i) {
            indices[i] = i;
        }

        std::partial_sort(
            indices.begin(), indices.begin() + 10, indices.end(),
            [&](std::size_t a, std::size_t b) {
                return last_logits[a] > last_logits[b];
            });

        std::cout << "SYCL logits range: ["
                  << *std::min_element(
                         last_logits.begin(), last_logits.end())
                  << ", "
                  << *std::max_element(
                         last_logits.begin(), last_logits.end())
                  << "]\n";

        std::cout << "Top 10 SYCL predictions:\n";
        for (std::size_t i = 0; i < 10; ++i) {
            const auto token = indices[i];
            std::cout << "  id=" << token
                      << " logit=" << last_logits[token] << '\n';
        }

        std::cout << "Saved " << V << " logits to "
                  << output_path << '\n';

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "GPT-2 logit export error: "
                  << e.what() << '\n';
        return 1;
    }
}
