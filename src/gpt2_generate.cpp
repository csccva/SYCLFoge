
/*
cmake -S . -B build -DCMAKE_CXX_COMPILER=icpx -DTINYML_BUILD_GPT2_GENERATE=ON;cmake --build build --target gpt2_generate -j2
 cmake -S . -B build -DCMAKE_CXX_COMPILER=icpx; cmake --build build --target gpt2_generate -j2
 ./build/gpt2_generate --max-new-tokens 20 "The capital of Finland is"
*/
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tinyml/tensor.hpp"
#include "tinyml/ops.hpp"
#include "tinyml/gpt2_model.hpp"
#include "tinyml/gpt2_tokenizer.hpp"

int main(int argc, char* argv[])
{
    try {
        // ------------------------------------------------------------
        // 1. Parse arguments
        //
        // Direct C++ mode:
        //   ./build/gpt2_generate --max-new-tokens 20 "Your prompt"
        //
        // Python-tokenizer mode:
        //   ./build/gpt2_generate --token-ids --max-new-tokens 20 464 3139
        // ------------------------------------------------------------
        int max_new_tokens = 20;
        bool token_ids_mode = false;
        std::string prompt;
        std::vector<std::int32_t> tokens;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];

            if (arg == "--token-ids") {
                if (!prompt.empty()) {
                    throw std::invalid_argument(
                        "Do not mix a text prompt with --token-ids");
                }
                token_ids_mode = true;
            }
            else if (arg == "--max-new-tokens") {
                if (i + 1 >= argc) {
                    throw std::invalid_argument(
                        "--max-new-tokens requires an integer");
                }

                std::size_t parsed = 0;
                const std::string value = argv[++i];
                max_new_tokens = std::stoi(value, &parsed);

                if (parsed != value.size()) {
                    throw std::invalid_argument(
                        "Invalid value for --max-new-tokens");
                }
            }
            else if (token_ids_mode) {
                std::size_t parsed = 0;
                const long long id = std::stoll(arg, &parsed);

                if (parsed != arg.size() ||
                    id < 0 ||
                    id >= static_cast<long long>(
                        tinyml::GPT2Model::VOCAB_SIZE)) {
                    throw std::invalid_argument(
                        "Invalid GPT-2 token ID: " + arg);
                }

                tokens.push_back(static_cast<std::int32_t>(id));
            }
            else {
                if (!prompt.empty()) {
                    prompt += ' ';
                }
                prompt += arg;
            }
        }

        if (max_new_tokens <= 0) {
            throw std::invalid_argument(
                "--max-new-tokens must be positive");
        }

        if (token_ids_mode) {
            if (tokens.empty()) {
                throw std::invalid_argument(
                    "--token-ids requires at least one token ID");
            }
        }
        else {
            if (prompt.empty()) {
                throw std::invalid_argument(
                    "Usage: gpt2_generate "
                    "[--max-new-tokens N] \"Your prompt\"");
            }
        }

        // ------------------------------------------------------------
        // 2. Tokenization
        // ------------------------------------------------------------
        tinyml::GPT2Tokenizer tokenizer(
            "models/gpt2/tokenizer/tokenizer.json");

        if (!token_ids_mode) {
            tokens = tokenizer.encode(prompt);
            std::cerr << "Prompt: " << prompt << '\n';
        }
        else {
            std::cerr << "Using token IDs supplied by the caller\n";
        }

        if (tokens.empty() ||
            tokens.size() > tinyml::GPT2Model::MAX_SEQ_LEN) {
            throw std::invalid_argument(
                "Prompt must contain between 1 and 1024 tokens");
        }

        if (static_cast<std::size_t>(max_new_tokens) >
            tinyml::GPT2Model::MAX_SEQ_LEN - tokens.size()) {
            throw std::invalid_argument(
                "Prompt plus generated tokens exceeds "
                "GPT-2's 1024-token context limit");
        }

        std::cerr << "Prompt tokens: " << tokens.size() << '\n';

        // ------------------------------------------------------------
        // 3. Initialize SYCL and load GPT-2
        // ------------------------------------------------------------
        sycl::queue queue{sycl::cpu_selector_v};

        std::cerr << "SYCL device: "
                  << queue.get_device().get_info<
                         sycl::info::device::name>()
                  << '\n';

        std::cerr << "Loading GPT-2 Small parameters...\n";

        tinyml::GPT2Model model(queue, "models/gpt2");

        std::vector<std::int32_t> generated;
        generated.reserve(static_cast<std::size_t>(max_new_tokens));

        // ------------------------------------------------------------
        // 4. Autoregressive generation using SYCL inference
        // ------------------------------------------------------------
        for (int step = 0; step < max_new_tokens; ++step) {
            const std::size_t L = tokens.size();
            const std::size_t vocab_size =
                tinyml::GPT2Model::VOCAB_SIZE;

            tinyml::Tensor<std::int32_t> token_ids(queue, {1, L});

            tinyml::Tensor<float> logits(
                queue, {1, L, vocab_size});

            // Copy token IDs to the device.
            auto e_input = ops::copy_to_device(
                queue, token_ids, tokens.data());

            e_input.wait_and_throw();

            // Run the existing GPT-2 model using SYCL.
            auto e_logits = model.forward(token_ids, logits);
            e_logits.wait_and_throw();

            // Transfer the final position's logits to the CPU.
            std::vector<float> last_logits(vocab_size);
            const std::size_t offset = (L - 1) * vocab_size;

            queue.memcpy(
                last_logits.data(),
                logits.data() + offset,
                vocab_size * sizeof(float)
            ).wait_and_throw();

            for (const float value : last_logits) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error(
                        "Non-finite logit encountered during generation");
                }
            }

            // Greedy decoding.
            const auto best = std::max_element(
                last_logits.begin(), last_logits.end());

            const auto next_id = static_cast<std::int32_t>(
                best - last_logits.begin());

            generated.push_back(next_id);
            tokens.push_back(next_id);

            std::cerr << "Generated token " << step + 1
                      << ": id=" << next_id
                      << ", logit=" << *best << '\n';

            // GPT-2 end-of-text token.
            if (next_id == 50256) {
                break;
            }
        }

        // ------------------------------------------------------------
        // 5. Output generated IDs and decoded text
        // ------------------------------------------------------------
        std::cout << "\nGENERATED_IDS:";
        for (const std::int32_t id : generated) {
            std::cout << ' ' << id;
        }
        std::cout << '\n';

        std::cout << "\nGenerated text:\n"
                  << tokenizer.decode(tokens)
                  << '\n';

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "GPT-2 generation error: "
                  << e.what() << '\n';
        return 1;
    }
}