/*
cmake -S tests -B build/test_tokenizer; cmake --build build/test_tokenizer -j2; ./build/test_tokenizer/test_gpt2_tokenizer
*/

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tinyml/gpt2_tokenizer.hpp"

int main()
{
    try {
        tinyml::GPT2Tokenizer tokenizer("models/gpt2/tokenizer.json");

        const std::vector<std::string> test_cases = {
            "Hello world!",
            "Hello  world",
            "It's a test.\nNext line.",
            "GPU optimization: 2× faster"
        };

        std::cout << "Vocabulary size: "
                  << tokenizer.vocab_size() << "\n\n";

        for (const auto& input : test_cases) {
            const auto ids = tokenizer.encode(input);
            const auto decoded = tokenizer.decode(ids);

            std::cout << "Input: " << input << '\n';
            std::cout << "Token IDs:";

            for (const auto id : ids) {
                std::cout << ' ' << id;
            }

            std::cout << "\nDecoded: " << decoded << "\n\n";

            if (decoded != input) {
                throw std::runtime_error(
                    "Tokenizer round-trip failed for input: " + input);
            }
        }

        std::cout << "All tokenizer tests passed.\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
